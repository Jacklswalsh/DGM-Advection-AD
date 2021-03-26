import numpy as np
from matplotlib import pyplot as plt

### Find ways to grade the elements ###
class Mesh():
    def mesh_gen(self, elements, start, end):

        # elements = 5
        # start = -8
        # end = 8

        mesh = []

        dx = (end-start)/(elements)

        x = np.arange(start, end, dx)
        # print(x)
        for i in x:
                mesh.append(np.array([i, i + (dx)], dtype='double'))

        return mesh

if __name__ == "__main__":

    mesh_obj = Mesh()
    mesh = mesh_obj.mesh_gen(4, -8, -3)
    mesh.extend(mesh_obj.mesh_gen(16, -3, 3))
    mesh.extend(mesh_obj.mesh_gen(4, 3, 8))

    # print(mesh)

    points = 5
    height = 0.6
    for i in (mesh):
        x = np.ones(points)*i[0]
        y = np.arange(0,height,(height/points))
        plt.plot(x,y, linestyle='dashed')
    plt.show()
