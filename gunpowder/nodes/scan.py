import logging
import multiprocessing
import numpy as np
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.producer_pool import ProducerPool
from gunpowder.volume import Volume
from gunpowder.volume_spec import VolumeSpec
from gunpowder.points import Points
from .batch_filter import BatchFilter

import h5py
import itertools

logger = logging.getLogger(__name__)

class Scan(BatchFilter):
    '''Iteratively requests batches of size ``reference`` from upstream
    providers in a scanning fashion, until all requested ROIs are covered. If
    the batch request to this node is empty, it will scan the complete upstream
    ROIs (and return nothing). Otherwise, it scans only the requested ROIs and
    returns a batch assembled of the smaller requests. In either case, the
    upstream requests will be contained in the downstream requested ROI or
    upstream ROIs.

    If mask_volume_type is not None, the Scan node will leave all batches empty where
    the roi of the VolumeType "relevant_type" is not within the ON region of the provided
    mask_volume_type. Batches with rois which are only partially in the ON region are
    processed normally.

    Args:

        reference(:class:`BatchRequest`): A reference :class:`BatchRequest`.
            This request will be shifted in a scanning fashion over the
            upstream ROIs of the requested volumes or points.

        num_workers (int, optional): If set to >1, upstream requests are made
            in parallel with that number of workers.

        cache_size (int, optional): If multiple workers are used, how many
            batches to hold at most.

        relevant_type (:class:`VolumeTypes`): VolumeType which's roi
            has to be at least partially covered by the mask for the batch to be considered.

        mask_volume_type (:class:`mask_volume_type`): VolumeType of mask which
            defines if the batch at shifted_reference is considered.

        path_to_mask (str): path to file with mask
            (in dvid format - resolution of block size 32).
    '''

    def __init__(self, reference, num_workers=1, cache_size=50, aggregate=True,
                 relevant_type=None, mask_volume_type=None, path_to_mask=None):

        self.reference = reference.copy()
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.aggregate = aggregate
        self.workers = None
        if num_workers > 1:
            self.request_queue = multiprocessing.Queue(maxsize=0)
        self.batch = None

        self.mask_volume_type = mask_volume_type
        self.relevant_type    = relevant_type
        self.path_to_mask = path_to_mask

    def setup(self):

        if self.num_workers > 1:
            self.workers = ProducerPool(
                [self.__worker_get_chunk for _ in range(self.num_workers)],
                queue_size=self.cache_size)
            self.workers.start()

    def teardown(self):

        if self.num_workers > 1:
            self.workers.stop()

    def provide(self, request):

        empty_request = (len(request) == 0)
        if empty_request:
            logger.warn('Scan got an empty request, will scan whole upstream provider')
            scan_spec = self.spec
        else:
            scan_spec = request

        stride = self.__get_stride()
        shift_roi = self.__get_shift_roi(scan_spec)

        shifts = self.__enumerate_shifts(shift_roi, stride)
        num_chunks = len(shifts)

        logger.info("scanning over %d chunks", num_chunks)

        # the batch to return
        self.batch = Batch()

        if self.num_workers > 1:
            counter = 0
            for shift in shifts:
                shifted_reference = self.__shift_request(self.reference, shift)

                if self.mask_volume_type is not None:
                    gt_mask_of_shifted_reference = self.__read_gt_mask(shifted_reference[self.relevant_type].roi)
                    if not gt_mask_of_shifted_reference.mean() == 0:
                        self.request_queue.put(shifted_reference)
                        counter += 1
                else:
                    self.request_queue.put(shifted_reference)
                    counter += 1


            for i in range(counter):  # num_chunks):

                chunk = self.workers.get()

                if not empty_request and self.aggregate:
                    self.__add_to_batch(request, chunk)

                logger.info("processed chunk %d/%d", i, num_chunks)

        else:

            for i, shift in enumerate(shifts):

                shifted_reference = self.__shift_request(self.reference, shift)

                if self.mask_volume_type is not None:
                    gt_mask_of_shifted_reference = self.__read_gt_mask(shifted_reference[self.relevant_type].roi)
                    if not gt_mask_of_shifted_reference.mean() == 0:
                        chunk = self.__get_chunk(shifted_reference)

                        if not empty_request and self.aggregate:
                            self.__add_to_batch(request, chunk)
                        logger.info("processed chunk %d/%d", i, num_chunks)

                else:
                    chunk = self.__get_chunk(shifted_reference)

                    if not empty_request and self.aggregate:
                        self.__add_to_batch(request, chunk)

                    logger.info("processed chunk %d/%d", i, num_chunks)

        # setup empty batch if no valid batches found at all
        if self.batch.get_total_roi() is None:
            chunk = self.__get_chunk(shifted_reference)
            self.batch = self.__setup_batch(request, chunk)

        batch = self.batch
        self.batch = None

        logger.debug("returning batch %s", batch)

        return batch

    def __get_stride(self):
        '''Get the maximal amount by which ``reference`` can be moved, such
        that it tiles the space.'''

        stride = None

        # get the least common multiple of all voxel sizes, we have to stride
        # at least that far
        lcm_voxel_size = self.spec.get_lcm_voxel_size(
            self.reference.volume_specs.keys())

        # that's just the minimal size in each dimension
        for identifier, reference_spec in self.reference.items():

            shape = reference_spec.roi.get_shape()

            for d in range(len(lcm_voxel_size)):
                assert shape[d] >= lcm_voxel_size[d], ("Shape of reference "
                                                       "ROI %s for %s is "
                                                       "smaller than least "
                                                       "common multiple of "
                                                       "voxel size "
                                                       "%s"%(reference_spec.roi,
                                                             identifier,
                                                             lcm_voxel_size))

            if stride is None:
                stride = shape
            else:
                stride = Coordinate((
                    min(a, b)
                    for a, b in zip(stride, shape)))

        return stride

    def __get_shift_roi(self, spec):
        '''Get the minimal and maximal shift (as a ROI) to apply to
        ``self.reference``, such that it is still fully contained in ``spec``.
        '''

        total_shift_roi = None

        # get individual shift ROIs and intersect them
        for identifier, reference_spec in self.reference.items():

            if spec[identifier].roi is None:
                continue

            # shift the spec roi such that its offset == shift from reference to
            # spec
            shift_roi = spec[identifier].roi.shift(-reference_spec.roi.get_offset())

            # shrink by the size of reference at the end
            shift_roi = shift_roi.grow(None, -reference_spec.roi.get_shape())

            if total_shift_roi is None:
                total_shift_roi = shift_roi
            else:
                total_shift_roi = total_shift_roi.intersect(shift_roi)
                if total_shift_roi.empty():
                    raise RuntimeError("There is no location where the ROIs "
                                       "the reference: %s \n are contained in the "
                                       "request/upstream ROIs: "
                                       "%s."%(self.reference, spec))

        if total_shift_roi is None:
            raise RuntimeError("None of the upstream ROIs are bounded (all "
                               "ROIs are None). Scan needs at least one "
                               "bounded upstream ROI.")

        return total_shift_roi

    def __enumerate_shifts(self, shift_roi, stride):
        '''Produces a sequence of shift coordinates starting at the beginning
        of ``shift_roi``, progressing with ``stride``. The maximum shift
        coordinate in any dimension will be the last point inside the shift roi
        in this dimension.'''

        min_shift = shift_roi.get_offset()
        max_shift = shift_roi.get_end()

        shift = np.array(min_shift)
        shifts = []

        dims = len(min_shift)

        logger.debug(
            "enumerating possible shifts of %s in %s", stride, shift_roi)

        while True:

            logger.debug("adding %s", shift)
            shifts.append(Coordinate(shift))

            if (shift == max_shift).all():
                break

            # count up dimensions
            for d in range(dims):

                if shift[d] >= max_shift[d]:
                    if d == dims - 1:
                        break
                    shift[d] = min_shift[d]
                else:
                    shift[d] += stride[d]
                    # snap to last possible shift, don't overshoot
                    if shift[d] > max_shift[d]:
                        shift[d] = max_shift[d]
                    break

        return shifts

    def __shift_request(self, request, shift):

        shifted = request.copy()
        for _, spec in shifted.items():
            spec.roi = spec.roi.shift(shift)

        return shifted

    def __worker_get_chunk(self):

        request = self.request_queue.get()
        return self.__get_chunk(request)

    def __get_chunk(self, request):

        return self.get_upstream_provider().request_batch(request)

    def __add_to_batch(self, spec, chunk):

        if self.batch.get_total_roi() is None:
            self.batch = self.__setup_batch(spec, chunk)

        for (volume_type, volume) in chunk.volumes.items():
            self.__fill(self.batch.volumes[volume_type].data, volume.data,
                        spec.volume_specs[volume_type].roi, volume.spec.roi,
                        self.spec[volume_type].voxel_size)

        for (points_type, points) in chunk.points.items():
            self.__fill_points(self.batch.points[points_type].data, points.data,
                               spec.points_specs[points_type].roi, points.roi)

    def __setup_batch(self, batch_spec, chunk):
        '''Allocate a batch matching the sizes of ``batch_spec``, using
        ``chunk`` as template.'''

        batch = Batch()

        for (volume_type, spec) in batch_spec.volume_specs.items():
            roi = spec.roi
            voxel_size = self.spec[volume_type].voxel_size

            # get the 'non-spatial' shape of the chunk-batch
            # and append the shape of the request to it
            volume = chunk.volumes[volume_type]
            shape = volume.data.shape[:-roi.dims()]
            shape += (roi.get_shape() // voxel_size)

            spec = self.spec[volume_type].copy()
            spec.roi = roi
            batch.volumes[volume_type] = Volume(data=np.zeros(shape),
                                                spec=spec)

        for (points_type, spec) in batch_spec.points_specs.items():
            roi = spec.roi
            spec = self.spec[points_type].copy()
            spec.roi = roi
            batch.points[points_type] = Points(data={}, spec=spec)

        logger.debug("setup batch to fill %s", batch)

        return batch

    def __fill(self, a, b, roi_a, roi_b, voxel_size):
        logger.debug("filling " + str(roi_b) + " into " + str(roi_a))

        roi_a = roi_a // voxel_size
        roi_b = roi_b // voxel_size

        common_roi = roi_a.intersect(roi_b)
        if common_roi is None:
            return

        common_in_a_roi = common_roi - roi_a.get_offset()
        common_in_b_roi = common_roi - roi_b.get_offset()

        slices_a = common_in_a_roi.get_bounding_box()
        slices_b = common_in_b_roi.get_bounding_box()

        if len(a.shape) > len(slices_a):
            slices_a = (slice(None),)*(len(a.shape) - len(slices_a)) + slices_a
            slices_b = (slice(None),)*(len(b.shape) - len(slices_b)) + slices_b

        a[slices_a] = b[slices_b]

    def __fill_points(self, a, b, roi_a, roi_b):
        logger.debug("filling points of " + str(roi_b) + " into points of" + str(roi_a))

        common_roi = roi_a.intersect(roi_b)
        if common_roi is None:
            return

        # find max point_id in a so far
        max_point_id = 0
        for point_id, point in a.items():
            if point_id > max_point_id:
                max_point_id = point_id

        for point_id, point in b.items():
            if roi_a.contains(Coordinate(point.location)):
                a[point_id + max_point_id] = point

    def __read_gt_mask(self, roi):

        voxel_size = self.spec[self.mask_volume_type].voxel_size

        block_size_vx = 32  # np.array([32, 32, 32], dtype='int')

        offset_in_vx         = np.asarray(roi.get_offset()/voxel_size)
        offset_in_blocks     = np.floor(offset_in_vx/block_size_vx).astype('int')
        offset_within_blocks = offset_in_vx % block_size_vx

        shape_in_vx            = np.asarray(roi.get_shape()/voxel_size)
        shape_in_vx_incl_offset_within_blocks = (shape_in_vx+offset_within_blocks).astype('float')
        shape_in_blocks        = np.ceil(np.array(shape_in_vx_incl_offset_within_blocks)/block_size_vx).astype('int')

        h5_file           = h5py.File(self.path_to_mask, 'r')
        gt_mask_in_blocks = h5_file['roi_binary_mask'][offset_in_blocks[0]:offset_in_blocks[0] + shape_in_blocks[0],
                                                   offset_in_blocks[1]:offset_in_blocks[1] + shape_in_blocks[1],
                                                   offset_in_blocks[2]:offset_in_blocks[2] + shape_in_blocks[2]]
        h5_file.close()

        if (np.unique(gt_mask_in_blocks) == 1).all():
            gt_mask = np.ones(shape_in_vx)

        else:
            gt_mask_in_vx = np.zeros(np.array(gt_mask_in_blocks.shape)*block_size_vx)
            for z, y, x in itertools.product(range(gt_mask_in_blocks.shape[0]), range(gt_mask_in_blocks.shape[1]), range(gt_mask_in_blocks.shape[2])):
                gt_mask_in_vx[z*block_size_vx:(z+1)*block_size_vx,
                              y*block_size_vx:(y+1)*block_size_vx,
                              x*block_size_vx:(x+1)*block_size_vx] = gt_mask_in_blocks[z,y,x]

            gt_mask = gt_mask_in_vx[offset_within_blocks[0]:offset_within_blocks[0]+shape_in_vx[0],
                      offset_within_blocks[1]:offset_within_blocks[1]+shape_in_vx[1],
                      offset_within_blocks[2]:offset_within_blocks[2]+shape_in_vx[2]]

        return gt_mask
