"""Inspect and edit a profiling database."""
import argparse

from alpa import DeviceCluster, ProfilingResultDatabase
from alpa.util import run_cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="/workspace/benchmark/harp/prof_database_A101.pkl")
    args = parser.parse_args()

    prof_database = ProfilingResultDatabase()
    prof_database.load(args.filename)

    # Do some editing
    new_data = {}
    for key, value in prof_database.data.items():
        # key[0] is the device name, key[1] is the mesh shape
        new_key = ("A101", key[1]) 
        new_data[new_key] = value
    prof_database.data = new_data
    prof_database.save(args.filename)
    
    
    for (_, submesh), value in prof_database.data.items():
        # prof_database.data[(_, submesh)].available_memory_per_device = 32 * 1e9
        print(f"Mesh: {submesh}, available_memory_per_device: {value.available_memory_per_device/1e9:.2f} GB, ")

    # Print results
    print("Meshes:")
    print(list(prof_database.data.keys()))
    print()

    mesh_result = prof_database.query("A100", (2, 2))
    print(mesh_result)

    mesh_result = prof_database.query("A100", (1, 2))
    print(mesh_result)

    mesh_result = prof_database.query("A100", (1, 1))
    print(mesh_result)

# new_data = {}
# for key, value in prof_database.data.items():
#     # key[0] is the device name, key[1] is the mesh shape
#     new_key = ("RTX4090", key[1]) 
#     new_data[new_key] = value

# # Update the dictionary in place
# prof_database.data = new_data