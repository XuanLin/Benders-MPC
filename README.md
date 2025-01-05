# Benders-MPC

Model Predictive Control implementation using Generalized Benders Decomposition (GBD), with applications to systems with contact dynamics.

## Overview
This project implements a model predictive control (MPC) framework using Generalized Benders Decomposition (GBD). The framework is particularly designed for handling systems with mixed-integer constraints, such as hybrid motion planning or contact dynamics. 

# Associated paper
This repo is associated with my paper titled [Accelerate Hybrid Model Predictive Control using Generalized Benders Decomposition](https://arxiv.org/pdf/2406.00780).

A more complete version is available [here](https://arxiv.org/pdf/2401.00917).

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

    %% Core solver classes at the top
    class BaseGBDSolver {
        #params_: SolverParams
        #master_problem_: unique_ptr~BaseMasterSolver~
        #subproblem_: unique_ptr~BaseSubSolver~
        +solve()
        #updateInitialConditions()
        #optimizeSubProblem()
        #solveMasterProblem()*
    }

    %% Core interfaces in the middle
    class BaseMasterSolver {
        #params_: SolverParams
        #in_param_: VectorDyn
        +updateInitialConditions()
        +solveMaster()*
        +addOptimalityCut()*
        +addFeasibilityCut()*
    }
    class BaseSubSolver {
        #params_: SolverParams
        #in_param_: VectorDyn
        #model_, model_infeas_: unique_ptr
        +optimize()
        +updateInitialConditions()
    }

    %% Implementations
    class GreedyMasterSolver {
        +solveMaster()
        +addOptimalityCut()
        +addFeasibilityCut()
    }
    class CartPoleSolver {
        -params_: CartPoleParams
    }

    %% Parameters/Utils on the right
    class SolverParams {
        +Q, Qn, R: MatrixDyn
        +E, F, G: MatrixDyn
        +H1, H2, H3: MatrixDyn
    }
    class CartPoleParams {
        +System-specific parameters
    }
    class FlyingRobotParams {
        +System-specific parameters
    }

    %% Relationships
    SolverParams <|-- CartPoleParams
    SolverParams <|-- FlyingRobotParams
    BaseMasterSolver <|-- GreedyMasterSolver
    BaseGBDSolver <|-- CartPoleSolver
    BaseGBDSolver o-- BaseMasterSolver : has
    BaseGBDSolver o-- BaseSubSolver : has
    CartPoleSolver ..> GreedyMasterSolver : uses by default
    CartPoleSolver ..> BaseSubSolver : uses by default
```
