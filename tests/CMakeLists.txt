add_executable(compile_test compile_test.cpp)
target_link_libraries(compile_test PRIVATE benders_mpc)
add_test(NAME compile_test COMMAND compile_test)

add_executable(test_free_flying_param test_free_flying_param.cpp)
target_link_libraries(test_free_flying_param PRIVATE benders_mpc)
add_test(NAME test_free_flying_param COMMAND test_free_flying_param)

# Configure testing
include(CTest)
set(CTEST_OUTPUT_ON_FAILURE ON)

