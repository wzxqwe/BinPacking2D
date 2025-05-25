from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import Domain

import sys
import math
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Preprocess import PreprocessBinPacking
from BinPackingData import *
from PlacementPoints import PlacementPointStrategy

from Model import *

class BinPackingSolverCP:
    def __init__(
        self, 
        items, 
        binDy, 
        binDx, 
        lowerBoundBins, 
        upperBoundBins, 
        placementPointStrategy = PlacementPointStrategy.UnitDiscretization, 
        timeLimit = 3600, 
        enableLogging = True,
        preprocess = None):

        self.items = items
        self.binDx = binDx
        self.binDy = binDy
        self.lowerBoundBins = lowerBoundBins
        self.upperBoundBins = upperBoundBins
        self.placementPointStrategy = placementPointStrategy
        self.timeLimit = timeLimit
        self.enableLogging = enableLogging

        self.preprocess = preprocess
        
        self.IsOptimal = False
        self.LB = -1
        self.UB = -1

        self.ItemBinAssignments = []
    
    def Solve(self, modelType = 'OneBigBin'):
        bin = Bin(self.binDx, self.binDy)

        if self.preprocess == None:
            self.preprocess = PreprocessBinPacking(self.items, bin, self.placementPointStrategy)
            self.preprocess.Run()

        if modelType == 'OneBigBin':
            model = OneBigBinModel(self.items, bin, self.preprocess, self.placementPointStrategy)
        elif modelType == 'StripPackOneBigBin':
            model = StripPackOneBigBinModel(self.items, bin, self.preprocess, self.placementPointStrategy)
        elif modelType == 'PairwiseAssignment':
            model = PairwiseAssignmentModel(self.items, bin, self.preprocess, self.placementPointStrategy)
        else:
            raise ValueError("Invalid bin packing model type.")
        
        rectangles = model.Solve(self.items, bin, self.lowerBoundBins, self.upperBoundBins, self.timeLimit, self.enableLogging)
       
        self.IsOptimal = model.IsOptimal
        self.LB = model.LB
        self.UB = model.UB
        self.ItemBinAssignments = model.ItemBinAssignments

        return rectangles

class BinPackBaseModel:
    def __init__(self, items, bin, preprocess, placementPointStrategy = PlacementPointStrategy.UnitDiscretization):
        self.IsOptimal = False
        self.LB = -1
        self.UB = 1

        self.placementPointStrategy = placementPointStrategy

        self.ItemBinAssignments = []

        self.solver = cp_model.CpSolver()
        self.model = cp_model.CpModel()

        self.startVariablesGlobalX = []
        self.startVariablesY = []
        self.intervalX = []
        self.intervalY = []
        self.rotateVariables = []
        self.effectiveWidth = []
        self.effectiveHeight = []
        self.endVariablesX = []
        self.endVariablesY = []

        self.itemArea = 0.0
        self.lowerBoundAreaBin = 0.0

        if preprocess == None:
            self.preprocess = PreprocessBinPacking(items, bin, placementPointStrategy)
        else:
            self.preprocess = preprocess

    def RetrieveLowerBound(self):
        return self.solver.BestObjectiveBound()

    def RetrieveUpperBound(self):
        return self.solver.ObjectiveValue()

    def CreateVariables(self, items, binDx, binDy):
        self.CreatePlacementVariables(items, binDx, binDy)
        self.CreateRotationVariables(items)
        self.CreateEffectiveDimensionVariables(items)
        self.CreateIntervalVariables(items)

    def CreatePlacementVariables(self, items, binDx, binDy):
        for i, item in enumerate(items):
            self.itemArea += item.Dx * item.Dy
            
            globalStartX = self.model.NewIntVar(0, binDx * self.preprocess.UpperBoundsBin, f'xb1.{i}')
            self.startVariablesGlobalX.append(globalStartX)

            yStart = self.model.NewIntVar(0, binDy, f'y1.{i}')
            self.startVariablesY.append(yStart)

    def CreateRotationVariables(self, items):
        for i in range(len(items)):
            rotate = self.model.NewBoolVar(f'rotate_{i}')
            self.rotateVariables.append(rotate)

    def CreateEffectiveDimensionVariables(self, items):
        for i, item in enumerate(items):
            max_dim = max(item.Dx, item.Dy)
            min_dim = min(item.Dx, item.Dy)
            effective_width = self.model.NewIntVar(min_dim, max_dim, f'eff_width_{i}')
            effective_height = self.model.NewIntVar(min_dim, max_dim, f'eff_height_{i}')
            
            self.model.Add(effective_width == item.Dx).OnlyEnforceIf(self.rotateVariables[i].Not())
            self.model.Add(effective_height == item.Dy).OnlyEnforceIf(self.rotateVariables[i].Not())
            self.model.Add(effective_width == item.Dy).OnlyEnforceIf(self.rotateVariables[i])
            self.model.Add(effective_height == item.Dx).OnlyEnforceIf(self.rotateVariables[i])
            
            self.effectiveWidth.append(effective_width)
            self.effectiveHeight.append(effective_height)

    def CreateIntervalVariables(self, items):
        for i in range(len(items)):
            end_x = self.model.NewIntVar(0, self.preprocess.Bin.Dx * self.preprocess.UpperBoundsBin, f'end_x_{i}')
            end_y = self.model.NewIntVar(0, self.preprocess.Bin.Dy, f'end_y_{i}')
            
            interval_x = self.model.NewIntervalVar(self.startVariablesGlobalX[i], self.effectiveWidth[i], end_x, f'xival{i}')
            interval_y = self.model.NewIntervalVar(self.startVariablesY[i], self.effectiveHeight[i], end_y, f'yival{i}')
            
            self.intervalX.append(interval_x)
            self.intervalY.append(interval_y)
            self.endVariablesX.append(end_x)
            self.endVariablesY.append(end_y)

    def CreateConstraints(self, items, binDx, binDy):
        self.model.AddNoOverlap2D(self.intervalX, self.intervalY)

        demandsY = [self.effectiveHeight[i] for i in range(len(items))]
        self.model.AddCumulative(self.intervalX, demandsY, binDy)
        
        for i in range(len(items)):
            self.model.Add(self.startVariablesY[i] + self.effectiveHeight[i] <= binDy)

    def SetParameters(self, solver, enableLogging, timeLimit):
        solver.parameters.log_search_progress = enableLogging
        solver.parameters.max_time_in_seconds = timeLimit
        solver.parameters.num_search_workers = 8

    def Solve(self, items, bin, lowerBoundBins, upperBoundBins, timeLimit = 3600, enableLogging = True):
        self.preprocess.Run()

        newItems = self.preprocess.ProcessedItems

        self.CreateVariables(newItems, bin.Dx, bin.Dy)
        self.CreateConstraints(newItems, bin.Dx, bin.Dy)
        self.CreateObjective(newItems, bin.Dx, bin.Dy)

        self.SetParameters(self.solver, enableLogging, timeLimit)
        self.SolveModel()

        rectangles = self.ExtractSolution(newItems, bin)

        return rectangles
    
    def CreateObjective(self, items, binDx, binDy):
        raise ValueError("CreateObjective() not implemented in BinPackBaseModel class.")

    def SolveModel(self):
        rc = self.solver.Solve(self.model)

    def ExtractSolution(self, items, bin):
        status = self.solver.StatusName()
        if status == 'UNKNOWN':
            raise ValueError("Start solution could not be determined (CP status == UNKNOWN)")

        self.IsOptimal = 1 if self.solver.StatusName() == 'OPTIMAL' else 0
        self.LB = self.RetrieveLowerBound()
        self.UB = self.RetrieveUpperBound()

        self.DetermineItemBinAssignments(items, bin)

        xArray = [self.solver.Value(self.startVariablesGlobalX[i]) for i in range(len(items))]
        yArray = [self.solver.Value(self.startVariablesY[i]) for i in range(len(items))]
        rotations = [self.solver.Value(self.rotateVariables[i]) for i in range(len(items))]
        widths = [self.solver.Value(self.effectiveWidth[i]) for i in range(len(items))]
        heights = [self.solver.Value(self.effectiveHeight[i]) for i in range(len(items))]

        rectangles = ExtractDataForPlotWithRotation(xArray, yArray, widths, heights, rotations, items, bin.Dx, bin.Dy)

        return rectangles

    def DetermineItemBinAssignments(self, items, bin):
        self.ItemBinAssignments = []
        for i in range(len(items)):
            binId = math.floor(self.solver.Value(self.startVariablesGlobalX[i]) / float(bin.Dx))
            self.ItemBinAssignments.append(binId)

class OneBigBinModel(BinPackBaseModel):
    def __init__(self, items, bin, preprocess, placementPointStrategy = PlacementPointStrategy.UnitDiscretization):
        super().__init__(items, bin, preprocess, placementPointStrategy)
        
        self.startVariablesLocalX = []
        self.binCountVariables = []
        self.placedBinVariables = []
        self.ItemBinAssignments = []

    def CreateVariables(self, items, binDx, binDy):
        super().CreateVariables(items, binDx, binDy)

        self.CreateLocalStartVariablesX(items, binDx)
        self.CreateItemBinAssignmentVariables(items)
        self.CreateBinCountVariables(binDx, binDy)

    def CreateLocalStartVariablesX(self, items, binDx):
        for i, item in enumerate(items):
            xStart = self.model.NewIntVar(0, binDx, f'x{i}')
            self.startVariablesLocalX.append(xStart)

    def CreateBinCountVariables(self, binDx, binDy):
        lowerBoundAreaBin = math.ceil(float(self.itemArea) / float(binDx * binDy))
        lowerBound = lowerBoundAreaBin

        self.binCountVariables = self.model.NewIntVar(lowerBound - 1, self.preprocess.UpperBoundsBin - 1, 'z')

    def CreateItemBinAssignmentVariables(self, items):
        binDomains = self.preprocess.BinDomains
        for i, item in enumerate(items):
            itemFeasibleBins = self.model.NewIntVarFromDomain(Domain.FromValues(binDomains[i]), f'b{i}')
            self.placedBinVariables.append(itemFeasibleBins)

    def CreateMaximumActiveBinConstraints(self, items):
        self.model.AddMaxEquality(self.binCountVariables, [self.placedBinVariables[i] for i in range(len(items))])

    def AddIncompatibilityCuts(self, incompatibleItems, fixItemToBin, model, binVariables):
        if incompatibleItems == None:
            return

        for i, j in incompatibleItems:
            if fixItemToBin[i] and fixItemToBin[j]:
                continue
            
            model.Add(binVariables[i] != binVariables[j])

    def CreateIntervalSynchronizationConstraints(self, items, binDx):
        for i, item in enumerate(items):
            self.model.Add(self.startVariablesGlobalX[i] == self.startVariablesLocalX[i] + self.placedBinVariables[i] * binDx)
            self.model.Add(self.startVariablesLocalX[i] + self.effectiveWidth[i] <= binDx)
    def CreateConstraints(self, items, binDx, binDy):
        super().CreateConstraints(items, binDx, binDy)

        self.CreateIntervalSynchronizationConstraints(items, binDx)
        self.CreateMaximumActiveBinConstraints(items)
        self.AddIncompatibilityCuts(self.preprocess.IncompatibleItems, self.preprocess.FixItemToBin, self.model, self.placedBinVariables)

    def CreateObjective(self, items, binDx, binDy):
        self.model.Minimize(self.binCountVariables + 1)

class StripPackOneBigBinModel(BinPackBaseModel):
    def __init__(self, items, bin, preprocess, placementPointStrategy = PlacementPointStrategy.UnitDiscretization):
        super().__init__(items, bin, preprocess, placementPointStrategy)

        self.loadingLength = None

    def CreateConstraints(self, items, binDx, binDy):
        self.model.AddNoOverlap2D(self.intervalX, self.intervalY)

        demandsY = [self.effectiveHeight[i] for i in range(len(items))]
        self.model.AddCumulative(self.intervalX, demandsY, binDy)
        
        for i in range(len(items)):
            self.model.Add(self.startVariablesY[i] + self.effectiveHeight[i] <= binDy)

    class EndPositionPlacementAbortCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, binDx):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.binDx = binDx
            self.LB = 1
            self.UB = binDx * 100

        def on_solution_callback(self):
            lb = self.BestObjectiveBound()
            ub = self.ObjectiveValue()

            self.LB = math.ceil(float(lb) / float(self.binDx))
            self.UB = math.ceil(float(ub) / float(self.binDx))
            if int(self.UB) - int(self.LB) == 0:
                self.StopSearch()

    def CreateObjective(self, items, binDx, binDy):
        lowerBoundAreaBin = math.ceil(float(self.itemArea) / float(binDy * binDx))

        self.loadingLength = self.model.NewIntVar((lowerBoundAreaBin - 1) * binDx + 1, (self.preprocess.UpperBoundsBin + 1) * binDx, 'z')

        self.model.AddMaxEquality(self.loadingLength, [self.intervalX[i].EndExpr() for i, item in enumerate(items)])

        self.model.Minimize(self.loadingLength)    

    def SolveModel(self):
        binDx = self.preprocess.Bin.Dx
        abortCallback = self.EndPositionPlacementAbortCallback(binDx)
        rc = self.solver.Solve(self.model, abortCallback)

    def RetrieveLowerBound(self):
        return math.ceil(float(self.solver.BestObjectiveBound()) / float(self.preprocess.Bin.Dx))

    def RetrieveUpperBound(self):
        return math.ceil(float(self.solver.ObjectiveValue()) / float(self.preprocess.Bin.Dx))

class PairwiseAssignmentModel(BinPackBaseModel):
    def __init__(self, items, bin, preprocess, placementPointStrategy = PlacementPointStrategy.UnitDiscretization):
        super().__init__(items, bin, preprocess, placementPointStrategy)

        self.itemBinAssignmentVariables = []
        self.binCountVariables = None

    def CreateVariables(self, items, binDx, binDy):
        self.CreatePlacementVariables(items, binDx, binDy)
        self.CreateRotationVariables(items)
        self.CreateEffectiveDimensionVariables(items)
        self.CreateIntervalVariables(items)
        self.CreateItemBinAssignmentVariables(items, self.preprocess.UpperBoundsBin)
        self.CreateBinCountVariables(binDx, binDy)

    def CreateItemBinAssignmentVariables(self, items, numberOfBins):
        self.itemBinAssignmentVariables = [[self.model.NewBoolVar(f'lit[{i}][{j}]') for j in range(numberOfBins)] for i in range(len(items))]

    def CreateBinCountVariables(self, binDx, binDy):
        lowerBoundAreaBin = math.ceil(float(self.itemArea) / float(binDx * binDy))
        lowerBound = lowerBoundAreaBin

        self.binCountVariables = self.model.NewIntVar(lowerBound, self.preprocess.UpperBoundsBin, 'z')

    def CreateConstraints(self, items, binDx, binDy):
        self.model.AddNoOverlap2D(self.intervalX, self.intervalY)

        demandsY = [self.effectiveHeight[i] for i in range(len(items))]
        self.model.AddCumulative(self.intervalX, demandsY, binDy)
        
        for i in range(len(items)):
            self.model.Add(self.startVariablesY[i] + self.effectiveHeight[i] <= binDy)

        self.CreateItemBinAssignmentConstraints(items, binDx, self.preprocess.UpperBoundsBin)

    def CreateItemBinAssignmentConstraints(self, items, binDx, numberOfBins):
        for i, item in enumerate(items):
            self.model.Add(sum(self.itemBinAssignmentVariables[i]) == 1)
            self.model.Add(sum(j * binDx * self.itemBinAssignmentVariables[i][j] for j in range(numberOfBins)) <= self.intervalX[i].StartExpr())
            self.model.Add(sum((j + 1) * binDx * self.itemBinAssignmentVariables[i][j] for j in range(numberOfBins)) >= self.intervalX[i].EndExpr())

    def CreateObjective(self, items, binDx, binDy):
        for i in range(len(items)):
            for j in range(self.preprocess.UpperBoundsBin):
                self.model.Add(self.binCountVariables >= j + 1).OnlyEnforceIf(self.itemBinAssignmentVariables[i][j])

        self.model.Minimize(self.binCountVariables)

def ExtractDataForPlotWithRotation(xArray, yArray, widths, heights, rotations, items, binDx, binDy):
    rectangles = []
    for i in range(len(items)):
        rect = {
            'x': xArray[i],
            'y': yArray[i],
            'width': widths[i],
            'height': heights[i],
            'rotated': rotations[i] == 1
        }
        rectangles.append(rect)
    return rectangles

def PlotSolution(maxW, H, binDx, rectangles):
    # Calculate number of bins
    bin_width = binDx  # Use bin width from input data
    num_bins = math.ceil(maxW / bin_width) if maxW > 0 else 1
    
    # Calculate utilization for each bin
    bin_areas = [0] * num_bins  # Total item area per bin
    bin_total_area = bin_width * H
    for rect in rectangles:
        bin_idx = math.floor(rect['x'] / bin_width)
        if 0 <= bin_idx < num_bins:
            item_area = rect['width'] * rect['height']
            bin_areas[bin_idx] += item_area
    utilizations = [100 * area / bin_total_area if bin_total_area > 0 else 0 for area in bin_areas]
    
    # Dynamically scale figure size based on total width and height
    max_dimension = max(maxW, H)
    scale_factor = 10 / max_dimension if max_dimension > 0 else 1  # Target 10 inches for largest dimension
    fig_width = maxW * scale_factor
    fig_height = H * scale_factor
    fig_width = max(6, min(20, fig_width))  # Ensure reasonable bounds
    fig_height = max(4, min(15, fig_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Draw each bin with clear boundaries
    for bin_idx in range(num_bins):
        bin_x = bin_idx * bin_width
        # Draw bin background with alternating colors
        ax.add_patch(Rectangle(
            (bin_x, 0),
            bin_width,
            H,
            facecolor='lightgray' if bin_idx % 2 == 0 else 'lightblue',
            edgecolor='black',
            linewidth=3,
            alpha=0.3
        ))
        # Label bin
        ax.text(
            bin_x + bin_width / 2,
            H + 2,
            f'Bin {bin_idx + 1}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Display utilizations in top-right corner
    utilization_text = '\n'.join(f'Bin {i+1}: {util:.2f}%' for i, util in enumerate(utilizations))
    ax.text(
        0.95, 0.95, utilization_text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    
    # Draw items
    num_items = len(rectangles)
    # Calculate average minimum dimension for font scaling
    avg_min_dim = sum(min(rect['width'], rect['height']) for rect in rectangles) / num_items if num_items > 0 else 1
    for i, rect in enumerate(rectangles):
        x = rect['x']
        y = rect['y']
        width = rect['width']
        height = rect['height']
        rotated = rect['rotated']
        
        # Color based on rotation
        color = 'blue' if rotated else 'green'
        ax.add_patch(Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        ))
        
        # Annotate item index and rotation status
        min_dim = min(width, height)
        fontsize = max(6, min(12, 10 * min_dim / avg_min_dim))  # Scale font based on item size
        ax.text(
            x + width / 2,
            y + height / 2,
            f'{i}\n{"R" if rotated else "NR"}',
            ha='center',
            va='center',
            fontsize=fontsize,
            color='white',
            fontweight='bold'
        )
    
    # Set plot limits and labels
    ax.set_xlim(0, maxW)
    ax.set_ylim(0, H + 4)  # Extra space for bin labels
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title(f'2D Bin Packing Solution ({num_bins} Bins)')
    
    # Add grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust aspect ratio to reflect actual proportions
    ax.set_aspect('equal')
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    plt.show()

def main():
    path = 'data/input/BPP/CLASS'
    instanceId = 9
    fileName = path 
    items, H, W = ReadBenchmarkData(path, str(instanceId) + '.json')

    solver = BinPackingSolverCP(items, H, W, 1, 1, PlacementPointStrategy.NormalPatterns, 200)
    print('upperboundxxx {}'.format(solver.upperBoundBins))
    rectangles = solver.Solve('OneBigBin')

    objBoundUB = solver.UB
    PlotSolution(objBoundUB * W, H, W, rectangles)

if __name__ == "__main__":
    main()