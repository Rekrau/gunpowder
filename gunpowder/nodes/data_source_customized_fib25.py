import numpy as np
import itertools
import json
import h5py

import logging
from copy import deepcopy

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import dvision
from gunpowder.points import PointsTypes, Points, PreSynPoint, PostSynPoint
from gunpowder.profiling import Timing

from gunpowder.points_spec import PointsSpec

from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class DvidSourceReadException(Exception):
    pass

class MaskNotProvidedException(Exception):
    pass

class DataInfo(object):
    def __init__(self, hostname=None, port=None, uuid=None, path_to_file=None):
        self.hostname = hostname
        self.port     = port
        self.url      = "http://{}:{}".format(self.hostname, self.port)
        self.uuid     = uuid
        self.path_to_file = path_to_file

class DataSourceCustomized_Fib25(BatchProvider):

    def __init__(self, raw_datainfo=None, labels_datainfo=None, gt_mask_datainfo=None, synapse_datainfo=None,
		         volume_array_names=None, volume_specs=None,
                 points_array_names={}, points_rois={}, points_voxel_size=None):
        """
        :param hostname: hostname for DVID server
        :type hostname: str
        :param port: port for DVID server
        :type port: int
        :param uuid: UUID of node on DVID server
        :type uuid: str
        :param volume_array_names: dict {VolumeTypes:  DVID data instance for data in VolumeTypes}
        :param points_voxel_size: (dict), :class:``PointsType`` to its voxel_size (tuple)
        """

        self.raw_datainfo     = raw_datainfo
        self.labels_datainfo  = labels_datainfo
        self.gt_mask_datainfo = gt_mask_datainfo
        self.synapse_datainfo = synapse_datainfo

        self.volume_array_names = volume_array_names
        self.volume_specs       = volume_specs

        self.points_array_names = points_array_names
        self.points_rois        = points_rois
        self.points_voxel_size  = points_voxel_size

        self.node_service = None
        self.dims = 0

        self.time_dict = {}
        self.time_dict[VolumeTypes.RAW]       = []
        self.time_dict[VolumeTypes.GT_MASK]   = []
        self.time_dict[VolumeTypes.GT_LABELS] = []

        if self.synapse_datainfo is not None:
            with open(self.synapse_datainfo.path_to_file, 'r') as f:
                self.syn_file_json = json.load(f)['data']

            # get max distance between pre & postsyn locations
            distances = []
            for annotation in self.syn_file_json:
                for partner in annotation['partners']:
                    distances.append(np.linalg.norm(np.asarray(annotation['T-bar']['location'])
                                                    - np.asarray(partner['location'])))
            self.max_distance_pre_post = np.max(distances)

    def setup(self):

        for volume_type in self.volume_specs.keys():
            self.provides(volume_type, self.volume_specs[volume_type])
        for points_type, points_name in self.points_array_names.items():
            self.provides(points_type, PointsSpec(roi=self.points_rois[points_type]))

        logger.info("DvidSource.spec:\n{}".format(self.spec))

    def get_spec(self):
        return self.spec

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()

        for (volume_type, vol_specs) in request.volume_specs.items():

            roi = vol_specs.roi
            # check if requested volumetype can be provided
            if volume_type not in spec:
                raise RuntimeError("Asked for %s which this source does not provide"%volume_type)
            # check if request roi lies within provided roi
            if not spec[volume_type].roi.contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s"%(volume_type, roi, spec[volume_type].ro))

            read = {
                VolumeTypes.RAW: self.__read_raw,
                VolumeTypes.GT_LABELS: self.__read_gt,
		        VolumeTypes.GT_MASK: self.__read_gt_mask,
            }[volume_type]

            logger.debug("Reading %s in %s..."%(volume_type, roi))
            batch.volumes[volume_type] = Volume(data=read(roi), spec=request[volume_type])

        # if pre and postsynaptic locations requested, their id : SynapseLocation dictionaries should be created
        # together s.t. the ids are unique and allow to find partner locations
        if PointsTypes.PRESYN in request.points_specs or PointsTypes.POSTSYN in request.points_specs:
            try:  # either both have the same roi, or only one of them is requested
                assert request.points_specs[PointsTypes.PRESYN].roi == request.points_specs[PointsTypes.POSTSYN].roi
            except:
                assert PointsTypes.PRESYN not in request.points_specs or PointsTypes.POSTSYN not in request.points_specs

            if PointsTypes.PRESYN in request.points_specs:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points_specs[PointsTypes.PRESYN].roi)
            elif PointsTypes.POSTSYN in request.points_specs:
                presyn_points, postsyn_points = self.__read_syn_points(roi=request.points_specs[PointsTypes.POSTSYN].roi)


        for (points_type, pts_spec) in request.points_specs.items():
            roi = pts_spec.roi
            # check if requested pointstype can be provided
            if points_type not in spec:
                raise RuntimeError("Asked for %s which this source does not provide"%points_type)
            # check if request roi lies within provided roi
            if not spec[points_type].roi.contains(roi):
                raise RuntimeError("%s's ROI %s outside of my ROI %s"%(points_type,roi,spec.points[points_type]))

            logger.debug("Reading %s in %s..."%(points_type, roi))
            id_to_point = {PointsTypes.PRESYN: presyn_points,
                           PointsTypes.POSTSYN: postsyn_points}[points_type]
            batch.points[points_type] = Points(data=id_to_point, spec=pts_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __get_roi(self, array_name, voxel_size):
        data_instance = dvision.DVIDDataInstance(self.hostname, self.port, self.uuid, array_name)
        info = data_instance.info
        roi_min = info['Extended']['MinPoint']
        if roi_min is not None:
            roi_min = Coordinate(roi_min[::-1])
        roi_max = info['Extended']['MaxPoint']
        if roi_max is not None:
            roi_max = Coordinate(roi_max[::-1])

        return Roi(offset=roi_min*voxel_size, shape=(roi_max - roi_min)*voxel_size)

    def __read_raw(self, roi):
        spec = self.get_spec()
        slices = (roi / spec[VolumeTypes.RAW].voxel_size).get_bounding_box()
        if self.raw_datainfo.path_to_file is not None:
            h5_file  = h5py.File(self.raw_datainfo.path_to_file, 'r')
            raw_data = h5_file['grayscale'][slices]
            h5_file.close()
            return raw_data
        else:
            data_instance = dvision.DVIDDataInstance(self.raw_datainfo.hostname, self.raw_datainfo.port,
                                                     self.raw_datainfo.uuid, self.volume_array_names[VolumeTypes.RAW])
            try:
                return data_instance[slices]
            except Exception as e:
                print(e)
                msg = "Failure reading raw at slices {} with {}".format(slices, repr(self))
                raise DvidSourceReadException(msg)

    def __read_gt(self, roi):
        spec = self.get_spec()
        slices = (roi/spec[VolumeTypes.GT_LABELS].voxel_size).get_bounding_box()
        if self.labels_datainfo.path_to_file is not None:
            h5_file  = h5py.File(self.labels_datainfo.path_to_file, 'r')
            data = h5_file['groundtruth'][slices]
            h5_file.close()
            return data
        else:
            data_instance = dvision.DVIDDataInstance(self.labels_datainfo.hostname, self.labels_datainfo.port,
                                                     self.labels_datainfo.uuid, self.volume_array_names[VolumeTypes.GT_LABELS])
            try:
                return data_instance[slices]
            except Exception as e:
                print(e)
                msg = "Failure reading GT at slices {} with {}".format(slices, repr(self))
                raise DvidSourceReadException(msg)

    def __read_gt_mask(self, roi):

        spec = self.get_spec()

        # h5_file = h5py.File(self.gt_mask_datainfo.path_to_file, 'r')
        # block_size_vx = h5_file['block_size'][:].astype('int')
        # h5_file.close()
        # block_size_vx np.array([32, 32, 32], dtype='int')
        # assert block_size_vx[0]==block_size_vx[1]==block_size_vx[2]

        block_size_vx = 32  # np.array([32, 32, 32], dtype='int')

        offset_in_vx         = np.asarray(roi.get_offset()/spec[VolumeTypes.GT_LABELS].voxel_size)
        offset_in_blocks     = np.floor(offset_in_vx/block_size_vx).astype('int')
        offset_within_blocks = offset_in_vx % block_size_vx

        shape_in_vx            = np.asarray(roi.get_shape()/spec[VolumeTypes.GT_LABELS].voxel_size)
        shape_in_vx_incl_offset_within_blocks = (shape_in_vx+offset_within_blocks).astype('float')
        shape_in_blocks        = np.ceil(np.array(shape_in_vx_incl_offset_within_blocks)/block_size_vx).astype('int')

        h5_file           = h5py.File(self.gt_mask_datainfo.path_to_file, 'r')
        gt_mask_in_blocks = h5_file['roi_binary_mask'][offset_in_blocks[0]:offset_in_blocks[0] + shape_in_blocks[0],
                                                   offset_in_blocks[1]:offset_in_blocks[1] + shape_in_blocks[1],
                                                   offset_in_blocks[2]:offset_in_blocks[2] + shape_in_blocks[2]]
        h5_file.close()

        if (np.unique(gt_mask_in_blocks) == 1).all():
            gt_mask = np.ones(shape_in_vx)

        else:
            gt_mask_in_vx = np.zeros(np.array(gt_mask_in_blocks.shape)*block_size_vx)
            for z, y, x in itertools.product(range(gt_mask_in_blocks.shape[0]), range(gt_mask_in_blocks.shape[1]),
                                             range(gt_mask_in_blocks.shape[2])):
                gt_mask_in_vx[z*block_size_vx:(z+1)*block_size_vx,
                              y*block_size_vx:(y+1)*block_size_vx,
                              x*block_size_vx:(x+1)*block_size_vx] = gt_mask_in_blocks[z,y,x]

            gt_mask = gt_mask_in_vx[offset_within_blocks[0]:offset_within_blocks[0]+shape_in_vx[0],
                      offset_within_blocks[1]:offset_within_blocks[1]+shape_in_vx[1],
                      offset_within_blocks[2]:offset_within_blocks[2]+shape_in_vx[2]]

        return gt_mask

    def __read_syn_points(self, roi):
        """ read json file from dvid source, in json format to create a PreSynPoint/PostSynPoint for every location given """

        if PointsTypes.PRESYN in self.points_voxel_size:
            voxel_size = self.points_voxel_size[PointsTypes.PRESYN]
        elif PointsTypes.POSTSYN in self.points_voxel_size:
            voxel_size = self.points_voxel_size[PointsTypes.POSTSYN]

        location_id = -1
        presyn_points_dict, postsyn_points_dict = {}, {}
        for synapse_id, annotation in enumerate(self.syn_file_json):
            syn_id = int(synapse_id)
            t_bar_location = np.asarray([annotation['T-bar']['location'][2],annotation['T-bar']['location'][1],
                                         annotation['T-bar']['location'][0]])*voxel_size

            if roi.contains(Coordinate(t_bar_location)):
                location_id = location_id + 1
                t_bar_location_id = location_id
                loc_id_of_presyn_partner = [t_bar_location_id]
            else:
                # if presyn location is far away from considered roi (more than max distance between any pre- & postsyn.
                # location in dataset) --> don't even check partners
                dim = len(annotation['T-bar']['location'])
                if not roi.grow(Coordinate((self.max_distance_pre_post,)*dim),
                            Coordinate((self.max_distance_pre_post,)*dim)).contains(Coordinate(t_bar_location)):
                    continue
                else:
                    loc_id_of_presyn_partner = []

            loc_ids_of_postsyn_partner = []
            for partner in annotation['partners']:
                partner_location = np.asarray([partner['location'][2], partner['location'][1], partner['location'][0]])*voxel_size
                if roi.contains(Coordinate(partner_location)):
                    location_id = location_id + 1
                    loc_ids_of_postsyn_partner.append(location_id)
                    syn_point = PostSynPoint(location=partner_location, location_id=location_id,
                                             synapse_id=syn_id, partner_ids=loc_id_of_presyn_partner, props={'body ID': partner['body ID']})
                    postsyn_points_dict[int(location_id)] = deepcopy(syn_point)

            if len(loc_id_of_presyn_partner)>0:
                syn_point = PreSynPoint(location=t_bar_location, location_id=t_bar_location_id,
                                        synapse_id=syn_id, partner_ids=loc_ids_of_postsyn_partner,
                                        props={'body ID': annotation['T-bar']['body ID']})
                presyn_points_dict[int(t_bar_location_id)] = deepcopy(syn_point)

        return presyn_points_dict, postsyn_points_dict


    def __repr__(self):
        return "DvidSource of Raw (hostname={}, port={}, uuid={}, raw_array_name={})".format(
            self.raw_datainfo.hostname, self.raw_datainfo.port, self.raw_datainfo.uuid,
		self.volume_array_names[VolumeTypes.RAW]) #  self.volume_array_names[VolumeTypes.GT_LABELS])
