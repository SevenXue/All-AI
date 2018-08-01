import h5py

def print_keras_weights(weight_file_path):
    f = h5py.File(weight_file_path)

    try:

        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")

        for key, value in f.attrs.items():
            print(" {}: {}".format(key, value))

        for layer, g in f.items():
            print(" {}".format(layer))
            print("  Attributes:")
            for key, value in g.attrs.items():
                print("{}: {}".format(key, value))

            print("Dataset:")
            for name, d in g.items():
                print("{}: {}".format(name, d))
                # print("{}: {}".format(name, d.value))
    finally:
        f.close()

if __name__ == '__main__':
    print_keras_weights('model/plan.h5')