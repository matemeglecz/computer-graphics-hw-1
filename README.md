# Computer Graphics first homework

## Specification
Create a program that displays a random graph aesthetically and allows the user to zoom in on any part of it while the remaining part is still visible. The graph consists of 50 nodes with a saturation of 5% (5% of possible edges are real edges). To achieve an aesthetic layout, the positions of the nodes are determined by both heuristics and a force-driven graph-fitting algorithm that follows the rules of the hyperbolic plane under the effect of pressing the SPACE button.

For focusing, the graph is arranged in the hyperbolic plane and projected onto the screen using the Beltrami-Klein method. Focusing is done by shifting the graph in the hyperbolic plane so that the part of interest is placed at the bottom of the hyperboloid. The visual projection of the offset is the difference between the momentary position of the right mouse button press and the momentary position of the mouse movement in the pressed state.

Each node is a circle of the hyperbolic plane with a texture identifying the node.

## Result
![elso](https://user-images.githubusercontent.com/58141904/160291190-41bdb66c-eee2-4eb0-8352-c9c4e9ec56c7.png)
![masodik](https://user-images.githubusercontent.com/58141904/160291193-0192ed13-3fec-4529-95f3-26bc79047dab.png)

