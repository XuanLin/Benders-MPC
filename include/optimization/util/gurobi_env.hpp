#pragma once
#include "gurobi_c++.h"

namespace optimization {
namespace util {

class GurobiEnv {
public:
    GurobiEnv() {
        env_.set("LogFile", "subQP.log");
        env_.start();
    }

    GRBEnv& get() { return env_; }

protected:
    GRBEnv env_{true};
};

} // namespace util
} // namespace optimization

