// dual_naming.hpp
#pragma once
#include <string>
#include <vector>
#include <algorithm>

namespace optimization {
namespace util {

struct DualNameManager {
    int N_, nx_, nu_, nc_;
    std::vector<std::string> dual_names_;
    std::vector<int> id_x0_;                       
    std::vector<std::vector<int>> id_x_dyn_;       
    std::vector<std::vector<int>> id_xuz_;         

    DualNameManager(int N, int nx, int nu, int nc)
        : N_(N), nx_(nx), nu_(nu), nc_(nc) 
    {
        // Reserve space for efficiency
        dual_names_.reserve((N + 1) * nx + N * nc);
        id_x0_.resize(nx);
        id_x_dyn_.resize(N, std::vector<int>(nx));
        id_xuz_.resize(N, std::vector<int>(nc));

        // Initial conditions
        for (int i_x = 0; i_x < nx_; i_x++) {
            dual_names_.push_back("x0_item_" + std::to_string(i_x));
        }

        // Dynamics
        for (int i_n = 0; i_n < N_; i_n++) {
            for (int i_x = 0; i_x < nx_; i_x++) {
                dual_names_.push_back("x_dyn_" + std::to_string(i_n) + "_item_" + std::to_string(i_x));
            }
        }

        // Control
        for (int i_n = 0; i_n < N_; i_n++) {
            for (int i_c = 0; i_c < nc_; i_c++) {
                dual_names_.push_back("xuz_" + std::to_string(i_n) + "_item_" + std::to_string(i_c));
            }
        }

        // Compute indices using find and distance
        for (int i_x = 0; i_x < nx_; i_x++) {
            ptrdiff_t pos = std::distance(dual_names_.begin(), 
                std::find(dual_names_.begin(), dual_names_.end(), "x0_item_" + std::to_string(i_x)));
            id_x0_[i_x] = pos;
        }

        for (int i_n = 0; i_n < N_; i_n++) {
            for (int i_x = 0; i_x < nx_; i_x++) {
                ptrdiff_t pos = std::distance(dual_names_.begin(),
                    std::find(dual_names_.begin(), dual_names_.end(), 
                        "x_dyn_" + std::to_string(i_n) + "_item_" + std::to_string(i_x)));
                id_x_dyn_[i_n][i_x] = pos;
            }
        }

        for (int i_n = 0; i_n < N_; i_n++) {
            for (int i_c = 0; i_c < nc_; i_c++) {
                ptrdiff_t pos = std::distance(dual_names_.begin(),
                    std::find(dual_names_.begin(), dual_names_.end(), 
                        "xuz_" + std::to_string(i_n) + "_item_" + std::to_string(i_c)));
                id_xuz_[i_n][i_c] = pos;
            }
        }
    }
};

} // namespace util
} // namespace optimization
