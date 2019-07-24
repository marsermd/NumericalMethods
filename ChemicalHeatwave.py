from enum import Enum

import numpy as np
import torch
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.animator import animate
from mayavi.tools.mlab_scene_model import MlabSceneModel
from traits.api import HasTraits, Range, Instance, Button, on_trait_change, Enum as EnumView
from traitsui.api import View, Item, HGroup, VGroup
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction

if __name__ == '__main__':
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if useCuda else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if useCuda else 'torch.FloatTensor')

def secondDeriviativeByPosition(tensor, dh):
    # Second deriviateive by poistion can be found by a simple difference scheme implemented as a convolution
    weights = torch.tensor([[[
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [1, -6.0, 1],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
    ]]]) / (dh ** 2)

    # Unfortunately, pytorch assumes that a convolution is run over multiple batches of multiple channels of data
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # 'Replicate' padding by 1 covers boundary conditions of a 3x3x3 convolution
    tensor = torch.nn.functional.pad(tensor, (1, 1, 1, 1, 1, 1), "replicate")
    result = torch.nn.functional.conv3d(input=tensor, weight=weights)

    return result.squeeze(0).squeeze(0)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


class HistoryNode:
    temperature = None
    concentration = None
    omega = None

    def __init__(self, temperature, concentration, w):
        self.temperature = temperature
        self.concentration = concentration
        self.omega = w


class VolumeViewType(Enum):
    TEMPERATURE = 0
    CONCENTRATION = 10

    MAX_OMEGA = 20
    MAX_TEMPERATURE = 30
    BURN_OUT_TIME = 40


class Simulation(HasTraits):
    """Reaction Order"""
    alpha = Range(0.5, 3.0,  0.5)
    time = Range(0.0, 500.0, 0.0)
    viewType = EnumView(*VolumeViewType)
    updateButton = Button("Update")

    # the layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=800, width=1000, show_label=False),
                VGroup(
                    HGroup(
                        "alpha", "viewType"
                    ),
                    "time",
                        "updateButton"
                    )
                )

    scene = Instance(MlabSceneModel, ())

    EPS = torch.tensor([1e-20])

    K = 4 * 1e7
    R = 8.3144598
    E = 9 * 1e4
    Q = 7e5
    T0 = 293
    DENSITY = 830
    C = 1990
    LAMBDA = 0.13
    D = 8e-12

    DELTA_T = Q / C
    TM = T0 + DELTA_T
    U = ((2 * K * LAMBDA) / (Q * DENSITY * DELTA_T) * (R * TM ** 2 / E) ** 2 * T0 / TM * np.exp(-E / (R * TM))) ** 0.5

    """From Zel'dovich, Frank-Kamenetskii"""
    BETA = R * TM / E
    """From Zel'dovich, Frank-Kamenetskii"""
    GAMMA = R * TM ** 2 / (E * DELTA_T)


    GRID_SIZE_X = 32
    GRID_SIZE_Y = 150
    GRID_SIZE_Z = 32
    GRID_SIZE = (GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z)
    dh = 2e-4
    dt = 1e-2


    history = None
    temperature = None
    concentration = None

    maxOmega = None
    maxTemperature = None
    burnOutTime = None

    viewTypeToTransferFunctions = None
    viewTypeToVolumeGetter = None

    def __init__(self):
        # Do not forget to call the parent's __init__
        super().__init__()
        k = self.LAMBDA / (self.DENSITY * self.C)
        print(f"beta = {self.BETA}, gamma={self.GAMMA}")
        print(f"LE = {self.D / k}")
        print(k)
        dh = k / self.U
        dx = self.BETA * dh
        dd = self.D / self.U
        dt = dx / self.U

        print(f"dx = {dx}, dd = {dd}, dt = {dt}")

        self.restartSimulation()

        source = self.scene.mlab.pipeline.scalar_field(self.temperature.cpu().numpy())
        volume = self.scene.mlab.pipeline.volume(source, vmin=0, vmax=self.TM * 2)
        volume.update_data()
        self.plot = volume

        self.onViewTypeChanged()
        volume._volume_property.shade = False

        self.scene.background = (0.2, 0.3, 0.5)

    def _getTemperatureTransferFunctions(self):
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(self.T0 + self.DELTA_T * 0.0, 0, 0, 0)
        ctf.add_rgb_point(self.T0 + self.DELTA_T * 0.7, 1, 0, 0)
        ctf.add_rgb_point(self.T0 + self.DELTA_T * 1.0, 1, 0.8, 0.2)
        ctf.add_rgb_point(self.T0 + self.DELTA_T * 1.1, 1, 0.8, 0.8)
        ctf.add_rgb_point(self.T0 + self.DELTA_T * 1.3, 1, 1, 1)

        otf = PiecewiseFunction()
        transparency = 0.3
        otf.add_point(self.T0 + self.DELTA_T * 0.0, 0.0 * transparency)
        otf.add_point(self.T0 + self.DELTA_T * 1.3, 1.0 * transparency)
        otf.use_log_scale = False

        return ctf, otf

    def _getConcentrationTransfereFunctions(self):
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(0, 0, 0, 0)
        ctf.add_rgb_point(1, 1, 1, 1)

        otf = PiecewiseFunction()
        transparency = 0.3
        otf.add_point(0, 1.0 * transparency)
        otf.add_point(1, 0.03 * transparency)
        otf.use_log_scale = False
        return ctf, otf

    def _getOmegaTransferFunctions(self):
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(0, 0, 0, 0)
        ctf.add_rgb_point(0.3, 0, 0, 1)
        ctf.add_rgb_point(0.6, 0, 1, 1)
        ctf.add_rgb_point(0.9, 1, 1, 1)

        otf = PiecewiseFunction()
        transparency = 0.3
        otf.add_point(0, 1.0 * transparency)
        otf.add_point(1.0, 1.0 * transparency)
        otf.use_log_scale = False
        return ctf, otf

    def _getTimeTransferFunctions(self):
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(0, 0, 0, 0)
        ctf.add_rgb_point(300, 1, 1, 1)

        otf = PiecewiseFunction()
        transparency = 0.3
        otf.add_point(0, 1.0 * transparency)
        otf.add_point(300, 1.0 * transparency)
        otf.use_log_scale = False
        return ctf, otf

    @on_trait_change('alpha,beta,gamma')
    def restartSimulation(self):
        self.history = []
        self.temperature = torch.full(self.GRID_SIZE, self.T0)
        self.concentration = torch.full(self.GRID_SIZE, 1)
        self.maxOmega = torch.full(self.GRID_SIZE, 0)
        self.maxTemperature = torch.full(self.GRID_SIZE, self.T0)
        self.burnOutTime = torch.full(self.GRID_SIZE, -1)
        self.applyTemperatureSource()

        self.viewTypeToTransferFunctions = {
            VolumeViewType.TEMPERATURE: self._getTemperatureTransferFunctions(),
            VolumeViewType.CONCENTRATION: self._getConcentrationTransfereFunctions(),
            VolumeViewType.MAX_TEMPERATURE: self._getTemperatureTransferFunctions(),
            VolumeViewType.MAX_OMEGA: self._getOmegaTransferFunctions(),
            VolumeViewType.BURN_OUT_TIME: self._getTimeTransferFunctions(),
        }

        self.viewTypeToVolumeGetter = {
            VolumeViewType.TEMPERATURE: lambda historyNode: historyNode.temperature,
            VolumeViewType.CONCENTRATION: lambda historyNode: historyNode.concentration,
            VolumeViewType.MAX_TEMPERATURE: lambda _: self.maxTemperature.cpu(),
            VolumeViewType.MAX_OMEGA: lambda _: self.maxOmega.cpu(),
            VolumeViewType.BURN_OUT_TIME: lambda _: self.burnOutTime.cpu(),
        }

    def getNewOmega(self):
        return -self.K * torch.pow(self.concentration, self.alpha) * torch.exp(-self.E / self.R / self.temperature)

    def getNewConcentration(self, omega):
        secondDeriviative = secondDeriviativeByPosition(self.concentration, self.dh)
        result = self.concentration + self.dt * (self.D * secondDeriviative + omega)
        return result * (result > 1e-20).float()

    def getNewTemperature(self, omega):
        k = self.LAMBDA / (self.DENSITY * self.C)
        secondDeriviative = secondDeriviativeByPosition(self.temperature, self.dh)
        result = self.temperature + self.dt * (k * secondDeriviative - omega * self.Q / self.C)
        return result * (result > 1e-20).float()

    def applyTemperatureSource(self):
        """Set heating from y direction"""
        idx = torch.Tensor([0]).long()
        self.temperature.index_fill_(1, idx, self.TM)

        # self.temperature = self.temperature.cpu()
        # for i in range(self.GRID_SIZE_X):
        #     for j in range(self.GRID_SIZE_Z):
        #         x = i / self.GRID_SIZE_X
        #         y = j / self.GRID_SIZE_Z
        #         if (x * 4 + y < 1) or (x * 4 + y > 4):
        #             self.temperature[i, 0, j] = self.TM
        #         #else:
        #         #    self.temperature[i, 0, j] = 0
        # self.temperature = self.temperature.cuda()

    def updateSimulation(self):
        omega = self.getNewOmega()
        newTemperature = self.getNewTemperature(omega)
        newConcentration = self.getNewConcentration(omega)

        self.temperature = newTemperature
        self.concentration = newConcentration

        self.applyTemperatureSource()

        self.maxTemperature = torch.max(self.maxTemperature, self.temperature)
        self.maxOmega = torch.max(self.maxOmega, -omega)

        if len(self.history) % 10 == 0:
            self.history.append(HistoryNode(self.temperature.cpu(), self.concentration.cpu(), omega.cpu()))
        else:
            self.history.append(self.history[-1])

    @on_trait_change('updateButton')
    def play(self):
        self.animateSimulation()

    @on_trait_change('viewType')
    def onViewTypeChanged(self):
        ctf, otf = self.viewTypeToTransferFunctions[self.viewType]

        self.plot._volume_property.set_color(ctf)
        self.plot._ctf = ctf
        self.plot.update_ctf = True

        self.plot._volume_property.set_scalar_opacity(otf)
        self.plot._otf = otf

        self.plot.update_data()

        self.update_view()

    @animate(delay=10)
    def animateSimulation(self):
        while True:
            self.time += 0.3
            yield

    @on_trait_change('time')
    def update_view(self):
        frame = int(self.time / self.dt)
        while len(self.history) <= frame:
            self.updateSimulation()

        historyNode = self.history[frame]

        values = self.viewTypeToVolumeGetter[self.viewType](historyNode)

        self.plot.mlab_source.trait_set(scalars=values.numpy())


if __name__ == '__main__':
    simulation = Simulation()
    simulation.configure_traits()