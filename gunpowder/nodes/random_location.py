import copy
import logging
from random import randint

import numpy as np
import json
from skimage.transform import integral_image, integrate
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.volume import VolumeTypes
from gunpowder.volume_spec import VolumeSpec
from gunpowder.points_spec import PointsSpec
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream
    provider.

    The random location is chosen such that the batch request roi lies entirely
    inside the provider's roi.

    If `min_masked` (and optionally `mask_volume_type`) are set, only
    batches are returned that have at least the given ratio of masked-in
    voxels. This is in general faster than using the ``Reject`` node, at the
    expense of storing an integral volume of the complete mask.

    If 'focus_points_type' is set, only batches are returned that have at least
    one point of focus_points_type within the roi of PointsTypes.focus_points_type.

    Remark
    ------
    focus_point_type does only work if there are only deterministic nodes upstream

    Args:

        min_masked: If non-zero, require that the random sample contains at
            least that ratio of masked-in voxels.

        mask_volume_type: The volume type to use for mask checks.

        focus_points_type: gunpowder.PointsTypes, PointsTypes considered when
            looking for good location of batch s.t. at least one point of this
            PointsTypes is contained in batch
    '''

    def __init__(self, min_masked=0, mask_volume_type=VolumeTypes.GT_MASK, focus_points_type=None,
                 path_to_dvid_rois=None):

        self.min_masked = min_masked
        self.mask_volume_type = mask_volume_type
        self.mask_spec = None
        self.focus_points_type = focus_points_type
        self.path_to_dvid_rois = path_to_dvid_rois

        if self.path_to_dvid_rois is not None:
            self.block_size_vx = 32.
            with open(self.path_to_dvid_rois, 'r') as f:
                json_file = json.load(f)
                self.dvid_rois = json_file['rois']  # list of [[z,y, x_start, x_end], ...]
                self.dvid_rois_probs = json_file['probs'] # prob dep on (x_end - x_start+1)

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()

        if self.upstream_roi is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        # clear bounding boxes of all provided volumes and points --
        # RandomLocation does not have limits (offsets are ignored)
        for identifier, spec in self.spec.items():
            spec.roi = None
            self.updates(identifier, spec)

    def prepare(self, request):

        shift_roi = None

        logger.debug("request: %s", request.volume_specs)
        logger.debug("my spec: %s", self.spec)

        if request.volume_specs.keys():
            lcm_voxel_size = self.spec.get_lcm_voxel_size(
                    request.volume_specs.keys())
            logger.debug(
                "restricting random locations to multiples of voxel size %s",
                lcm_voxel_size)
        else:
            lcm_voxel_size = None

        for identifier, spec in request.items():

            request_roi = spec.roi
            if identifier in self.upstream_spec:
                provided_roi = self.upstream_spec[identifier].roi
            else:
                raise Exception(
                    "Requested %s, but upstream does not provide "
                    "it."%identifier)
            type_shift_roi = provided_roi.shift(-request_roi.get_begin()).grow((0,0,0),-request_roi.get_shape())

            if shift_roi is None:
                shift_roi = type_shift_roi
            else:
                shift_roi = shift_roi.intersect(type_shift_roi)

        logger.debug("valid shifts for request in " + str(shift_roi))

        assert shift_roi is not None and shift_roi.size() > 0, (
                "Shift Roi (%s) cannot satisfy batch request, no location covers all "
                "requested ROIs."%(shift_roi))

        if lcm_voxel_size is not None:
            lcm_shift_roi = shift_roi/lcm_voxel_size
        else:
            lcm_shift_roi = shift_roi


        block_size_nm = self.block_size_vx * np.array(lcm_voxel_size)
        good_location_found_for_mask, good_location_found_for_points = False, False
        while not good_location_found_for_mask or not good_location_found_for_points:

            if self.path_to_dvid_rois is not None:
                # select a random block of the officially correct rois, with probability prop to the roi's size
                valid_roi_found = False
                while not valid_roi_found:
                    chosen_block        = self.dvid_rois[int(np.random.choice(a=range(len(self.dvid_rois)), size=1,
                                                                              p=self.dvid_rois_probs))]
                    chosen_block_as_roi = Roi(offset=(chosen_block[0]*block_size_nm[0], chosen_block[1]*block_size_nm[1],
                                                      chosen_block[2]*block_size_nm[2]),
                                              shape=(block_size_nm[0], block_size_nm[1],
                                                      (chosen_block[3]-chosen_block[2]+1)*block_size_nm[2]))

                    #TOOO: Does not work if not focus_points_given...
                    # assert self.focus_points_type is not None, 'Script does not work properly if no points to focus on provided'
                    roi_focused_points = self.upstream_spec[self.focus_points_type].roi
                    if roi_focused_points.grow(Coordinate((0,0,0)), -request[self.focus_points_type].roi.get_shape()).contains(chosen_block_as_roi):
                        valid_roi_found = True

                random_shift = Coordinate(
                        randint(int(begin), int(end-1))
                        for begin, end in zip(chosen_block_as_roi.get_begin(), chosen_block_as_roi.get_end())) -\
                               request[self.focus_points_type].roi.get_offset()

            else:
                # select a random point inside ROI
                random_shift = Coordinate(
                        randint(int(begin), int(end-1))
                        for begin, end in zip(lcm_shift_roi.get_begin(), lcm_shift_roi.get_end()))
                if lcm_voxel_size is not None:
                    random_shift *= lcm_voxel_size

            initial_random_shift = copy.deepcopy(random_shift)
            logger.debug("random shift: " + str(random_shift))

            good_location_found_for_mask, good_location_found_for_points = False, False
            if self.focus_points_type is not None:

                focused_points_roi    = request[self.focus_points_type].roi
                focused_points_offset = focused_points_roi.get_offset()
                focused_points_shape  = focused_points_roi.get_shape()

                # prefetch points in roi of focus_points_type
                request_for_focused_pointstype = BatchRequest()
                request_for_focused_pointstype[self.focus_points_type] = PointsSpec(roi=focused_points_roi.shift(random_shift))

                # NOTE: If any nodes are placed between this node and the source
                # this function call might cause bugs
                batch_of_points = self.get_upstream_provider().request_batch(request_for_focused_pointstype)
                point_ids_in_batch = batch_of_points.points[self.focus_points_type].data.keys()

                if len(point_ids_in_batch) > 0:
                    chosen_point_id       = np.random.choice(point_ids_in_batch, size=1)[0]
                    chosen_point_location = Coordinate(batch_of_points.points[self.focus_points_type].data[chosen_point_id].location)
                    distance_focused_roi_to_chosen_point = chosen_point_location - (initial_random_shift + focused_points_offset)
                    local_shift_roi = Roi(offset=(distance_focused_roi_to_chosen_point - (focused_points_shape - Coordinate((2,2,2)))),
                                          shape=(focused_points_shape-Coordinate((2,2,2))))

                    # set max trials to prevent endless loop and search for suitable local shift in impossible cases
                    trial_nr, max_trials, good_local_shift = 0, 100000, False
                    while not good_local_shift:
                        trial_nr += 1
                        local_shift = Coordinate(
                                    randint(int(begin), int(end))
                                    for begin, end in zip(local_shift_roi.get_begin(), local_shift_roi.get_end()))
                        # make sure that new shift matches ROIs of all requested volumes
                        if shift_roi.contains(initial_random_shift + local_shift):
                            random_shift = initial_random_shift + local_shift
                            assert Roi(offset=random_shift + focused_points_offset,
                                        shape=focused_points_shape).contains(chosen_point_location)
                            good_local_shift, good_location_found_for_points = True, True

                        if trial_nr == max_trials:
                            good_location_found_for_points = False
                            break
                else:
                    good_location_found_for_points = False
            else:
                good_location_found_for_points = True

            if self.min_masked > 0:

                mask_voxel_size = self.spec[self.mask_volume_type].voxel_size

                # prefetch gt mask in roi of focus_points_type
                request_for_gt_mask = BatchRequest()
                request_for_gt_mask[self.mask_volume_type] = VolumeSpec(roi=focused_points_roi.shift(random_shift),\
                                                             voxel_size=mask_voxel_size, interpolatable=False, dtype=None)

                batch_of_gt_mask = self.get_upstream_provider().request_batch(request_for_gt_mask)
                mask_ratio = batch_of_gt_mask.volumes[self.mask_volume_type].data.mean()
                logger.debug("mask ratio is %f", mask_ratio)

                if mask_ratio >= self.min_masked:
                    logger.debug("good batch found")
                    good_location_found_for_mask = True
                else:
                    logger.debug("bad batch found")

            else:
                good_location_found_for_mask = True

        # shift request ROIs
        self.random_shift = random_shift
        for specs_type in [request.volume_specs, request.points_specs]:
            for (type, spec) in specs_type.items():
                roi = spec.roi.shift(random_shift)
                logger.debug("new %s ROI: %s"%(type, roi))
                specs_type[type].roi = roi
                assert self.upstream_roi.contains(roi)


    def process(self, batch, request):
        # reset ROIs to request
        for (volume_type, spec) in request.volume_specs.items():
            batch.volumes[volume_type].spec.roi = spec.roi
        for (points_type, spec) in request.points_specs.items():
            batch.points[points_type].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for points_type in request.points_specs.keys():
            for point_id, point in batch.points[points_type].data.items():
                batch.points[points_type].data[point_id].location -= self.random_shift

    def lcm(self, a, b):

        return
