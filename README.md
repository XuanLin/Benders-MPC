# Benders-MPC

Model Predictive Control implementation using Generalized Benders Decomposition (GBD), with applications to systems with contact dynamics.

## Overview
This project implements a model predictive control (MPC) framework using Generalized Benders Decomposition (GBD). The framework is particularly designed for handling systems with mixed-integer constraints, such as hybrid motion planning or contact dynamics. 

## Demonstrations

<div align="center">

| Cart Pole | Humanoid Balancing |
|:---------:|:------------------:|
| <img src="docs/media/cart_pole.gif" height="300" alt="Cart Pole Demo"/> | <img src="docs/media/humanoid_balancing.gif" height="300" alt="Humanoid Balancing Demo"/> |

</div>

# Associated paper
This repo is associated with my paper titled [Accelerate Hybrid Model Predictive Control using Generalized Benders Decomposition](https://arxiv.org/pdf/2406.00780).

## Key Features
- Generalized Benders Decomposition framework for MPC problems
- Example implementations:
  - Cart-pole system with wall contacts
  - Free-flying robot with obstacle avoidance
  - Motion planning through a dynamic maze
- Python bindings for simulation and visualization
- Integration with Gurobi optimizer

## Prerequisites
- CMake (3.10 or higher)
- C++ compiler with C++17 support
- Python 3.8 or later
- Gurobi Optimizer
- PyBullet (for simulation)

## Building
```bash
mkdir build && cd build
cmake ..
make
```

## Framework Architecture
The GBD framework is structured with the following key components:

```mermaid
classDiagram
    direction TB
    
    %% Base/Abstract Classes Layer
    class BaseGBDSolver {
        #params_: SolverParams
        #master_problem_: ~BaseMasterSolver~
        #sub_problem_: ~BaseSubSolver~
        +solve()
        #updateInitialConditions()
        #solveSubProblem()*
        #solveMasterProblem()*
    }
    class BaseMasterSolver {
        #params_: SolverParams 
        #in_param_: VectorDyn
        +updateInitialConditions()
        +solveMaster()*
        +addOptimalityCut()*
        +addFeasibilityCut()*
        +storeOptimalityCut()*
        +storeFeasibilityCut()*
    }
    class BaseSubSolver {
        #params_: SolverParams
        #in_param_: VectorDyn
        +updateInitialConditions()
        +solveSub()*
    }

    %% Implementation Layer
    class GurobiSubSolver {
        -model_: ~GRBModel~
        -model_infeas_: ~GRBModel~
        +solveSub()
    }
    class GreedyMasterSolver {
        +solveMaster()
        +addOptimalityCut()
        +addFeasibilityCut()
        +storeOptimalityCut()
        +storeFeasibilityCut()
    }
    class CartPoleSolver {
        -params_: CartPoleParams
    }

    %% Parameters Layer (moved to right)
    class SolverParams {
        +Q, Qn, R: MatrixDyn
        +E, F, G: MatrixDyn
        +H1, H2, H3: MatrixDyn
    }
    class CartPoleParams {
    }
    class FlyingRobotParams {
    }

    %% Inheritance Relationships
    SolverParams <|-- CartPoleParams
    SolverParams <|-- FlyingRobotParams
    BaseMasterSolver <|-- GreedyMasterSolver
    BaseSubSolver <|-- GurobiSubSolver
    BaseGBDSolver <|-- CartPoleSolver

    %% Composition Relationships
    BaseGBDSolver o-- BaseMasterSolver : has
    BaseGBDSolver o-- BaseSubSolver : has

    %% Dependencies
    CartPoleSolver ..> GreedyMasterSolver : uses by default
    CartPoleSolver ..> GurobiSubSolver : uses by default
    CartPoleSolver ..> CartPoleParams : uses
```
