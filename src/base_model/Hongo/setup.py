from setuptools import Extension, setup

## run python setup.py build_ext --inplace

source_dir = "../src"
hongo_module = Extension(
    "_hongo",
    sources=[
        "./CMakeFiles/hongo.dir/SimulationPYTHON_wrap.cxx",
        f"{source_dir}/Agent.cpp",
        f"{source_dir}/Bus.cpp",
        f"{source_dir}/ChoiceSet.cpp",
        f"{source_dir}/Dijkstra.cpp",
        f"{source_dir}/Kepler.cpp",
        f"{source_dir}/Lane.cpp",
        f"{source_dir}/Link.cpp",
        f"{source_dir}/Node.cpp",
        f"{source_dir}/Pedestrian.cpp",
        f"{source_dir}/Physarum.cpp",
        f"{source_dir}/RL.cpp",
        f"{source_dir}/Signal.cpp",
        f"{source_dir}/Simulation.cpp",
        f"{source_dir}/SparseMat.cpp",
        f"{source_dir}/Station.cpp",
        f"{source_dir}/Table.cpp",
        f"{source_dir}/Utility.cpp",
        f"{source_dir}/Vehicle.cpp",
    ],
    include_dirs=[f"{source_dir}", "/usr/include/python3.9"],
    extra_compile_args=["-std=c++17"],
)

setup(
    name="hongo",
    version="0.1",
    author="dogawa",
    description="""Hongo simulation module""",
    ext_modules=[hongo_module],
    py_modules=["hongo"],
)
