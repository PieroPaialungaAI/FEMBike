# Finite Element Method applied to a bike structure. 

n this project, a bike body has been studied using a Finite Element Method
(FEM) simulation.
This work has been made by Natalie Reed an Piero Paialunga, which have
worked together in the same office and room for the entire project. This
means that all the steps of the project, from the code developing to the re-
port writing, have been made together in the same room.
Starting, from scratch and having a bike model as a guideline, we created the
bike structure given a certain number of node N = 158. From this simple
structure, and having the model of the mechanical properties of the bike, we
created the stiffness and mass matrix. Moreover, the transformation matrix
(to change the coordinates of the stiffness and mass matrix) has been created
as well. Given that we consider the two wheels of the bike to be steady, we
applied boundary conditions there, considering 0 displacement on the three
directions. Two different static scenario have been considered. The first one
is a person sitting on the bike, the second one is a person sitting on the same
point of the bike but pushing the bike forward on the x directions. Given
these two forces, and applying a numerical solver, we found and displayed
the displacement on the entire bike and the resulting internal forces. The
dynamic problem has been started as well, but as it required a non trivial
numerical implementation given the necessity of building a differential equa-
tion solver, it has not been fully implemented.
The whole code has been built from scratch using Python. The entire struc-
ture of the code is parametric. This means that virtually infinite scenarios
can be considered by just changing the number of nodes, the mechanical prop-
erties, the boundary conditions and the force vectors in the correspondent
input .csv files. The code has been proven to be correct by being applied
in a controlled situation where we knew the exact displacement results. In
our study case, the results have furnished insightful considerations about the
displacement on the entire bike structure. An interesting development of this
project would be to compare this result with the ones of an FEM software
like ABACUS or to implement a system of differential equation solver to
solve the dynamic scenario that has been here prepared to be solved.
