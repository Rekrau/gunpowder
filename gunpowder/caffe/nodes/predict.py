import copy
import logging
import os

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.generic_predict import GenericPredict
from gunpowder.array import ArrayKey, Array


logger = logging.getLogger(__name__)

class Predict(GenericPredict):
    '''Augments a batch with network predictions.

    Args:

        prototxt (string): Filename of the network prototxt.

        weights (string): Filename of the network weights.

        inputs (dict): Dictionary from the names of input layers in the
            network to :class:``ArrayKey`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output layers in the
            network to :class:``ArrayKey``. New arrays will be generated by
            this node for each entry (if requested downstream).

        array_specs (dict, optional): An optional dictionary of
            :class:`ArrayKey` to :class:`ArraySpec` to set the array specs
            generated arrays (``outputs``). This is useful to set the
            ``voxel_size``, for example, if they differ from the voxel size of
            the input arrays. Only fields that are not ``None`` in the given
            :class:`ArraySpec` will be used.

        use_gpu (int): Which GPU to use. Set to ``None`` for CPU mode.
    '''

    def __init__(
            self,
            prototxt,
            weights,
            inputs,
            outputs,
            array_specs=None,
            use_gpu=None):

        super(Predict, self).__init__(
            inputs,
            outputs,
            array_specs,
            spawn_subprocess=True)
        for f in [prototxt, weights]:
            if not os.path.isfile(f):
                raise RuntimeError("%s does not exist"%f)
        self.prototxt = prototxt
        self.weights = weights
        self.inputs = inputs
        self.outputs = outputs
	self.use_gpu = use_gpu

    def start(self):

        logger.info("Initializing solver...")

        if self.use_gpu is not None:

            logger.debug("Predict process: using GPU %d"%self.use_gpu)
            caffe.enumerate_devices(False)
            caffe.set_devices((self.use_gpu,))
            caffe.set_mode_gpu()
            caffe.select_device(self.use_gpu, False)

        self.net = caffe.Net(self.prototxt, self.weights, caffe.TEST)
        self.net_io = NetIoWrapper(self.net, self.outputs.keys())

    def predict(self, batch, request):

        self.net_io.set_inputs({
            input_name: batch.arrays[array_key].data
            for input_name, array_key in self.inputs.items()
        })

        self.net.forward()
        output = self.net_io.get_outputs()

        for output_name, array_key in self.outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                    output[output_name][0], # strip #batch dimension
                    spec)

        return batch
