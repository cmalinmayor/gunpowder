import logging
import multiprocessing
import time

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied, NoResult
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec

logger = logging.getLogger(__name__)

class PredictProcessDied(Exception):
    pass

class GenericPredict(BatchFilter):
    '''Generic predict node to add predictions of a trained network to each each
    batch that passes through. This node alone does nothing and should be
    subclassed for concrete implementations.

    Args:

        inputs (dict): Dictionary from names of input layers in the network to
            :class:``ArrayKey`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output layers in the
            network to :class:``ArrayKey``. New arrays will be generated by
            this node for each entry (if requested downstream).

        array_specs (dict, optional): An optional dictionary of
            :class:`ArrayKey` to :class:`ArraySpec` to set the array specs
            generated arrays (``outputs`` and ``gradients``). This is useful
            to set the ``voxel_size``, for example, if they differ from the
            voxel size of the input arrays. Only fields that are not ``None``
            in the given :class:`ArraySpec` will be used.

        spawn_subprocess (bool, optional): Whether to run ``predict`` in a
            separate process. Default is false.
    '''

    def __init__(
            self,
            inputs,
            outputs,
            array_specs=None,
            spawn_subprocess=False):

        self.initialized = False

        self.inputs = inputs
        self.outputs = outputs
        self.array_specs = {} if array_specs is None else array_specs
        self.spawn_subprocess = spawn_subprocess

        if self.spawn_subprocess:

            # start prediction as a producer pool, so that we can gracefully
            # exit if anything goes wrong
            self.worker = ProducerPool([self.__produce_predict_batch], queue_size=1)
            self.batch_in = multiprocessing.Queue(maxsize=1)
            self.predict_lock = multiprocessing.Lock()

    def setup(self):

        # get common voxel size of inputs, or None if they differ
        common_voxel_size = None
        for key in self.inputs.values():

            if not isinstance(key, ArrayKey):
                continue

            voxel_size = self.spec[key].voxel_size

            if common_voxel_size is None:
                common_voxel_size = voxel_size
            elif common_voxel_size != voxel_size:
                common_voxel_size = None
                break

        # announce provided outputs
        for key in self.outputs.values():

            if key in self.array_specs:
                spec = self.array_specs[key].copy()
            else:
                spec = ArraySpec()

            if spec.voxel_size is None:

                assert common_voxel_size is not None, (
                    "There is no common voxel size of the inputs, and no "
                    "ArraySpec has been given for %s that defines "
                    "voxel_size."%key)

                spec.voxel_size = common_voxel_size

            if spec.interpolatable is None:

                # default for predictions
                spec.interpolatable = False

            self.provides(key, spec)

        if self.spawn_subprocess:
            self.worker.start()

    def teardown(self):
        if self.spawn_subprocess:
            # signal "stop"
            self.batch_in.put((None, None))
            try:
                self.worker.get(timeout=2)
            except NoResult:
                pass
            self.worker.stop()
        else:
            self.stop()

    def prepare(self, request):

        if not self.initialized and not self.spawn_subprocess:
            self.start()
            self.initialized = True

    def process(self, batch, request):

        if self.spawn_subprocess:

            with self.predict_lock:

                self.batch_in.put((batch, request))

                try:
                    out = self.worker.get()
                except WorkersDied:
                    raise PredictProcessDied()

            for array_key in self.outputs.values():
                if array_key in request:
                    batch.arrays[array_key] = out.arrays[array_key]

        else:

            self.predict(batch, request)

    def start(self):
        '''To be implemented in subclasses.

        This method will be called before the first call to :fun:`predict`,
        from the same process that :fun:`predict` will be called from. Use
        this to initialize your model and hardware.
        '''
        pass

    def predict(self, batch, request):
        '''To be implemented in subclasses.

        In this method, an implementation should predict arrays on the given
        batch. Output arrays should be created according to the given request
        and added to ``batch``.'''
        raise NotImplementedError("Class %s does not implement 'predict'"%self.name())

    def stop(self):
        '''To be implemented in subclasses.

        This method will be called after the last call to :fun:`predict`,
        from the same process that :fun:`predict` will be called from. Use
        this to tear down your model and free training hardware.
        '''
        pass

    def __produce_predict_batch(self):
        '''Process one batch.'''

        if not self.initialized:

            self.start()
            self.initialized = True

        batch, request = self.batch_in.get()

        # stop signal
        if batch is None:
            self.stop()
            return None

        self.predict(batch, request)

        return batch

