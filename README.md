# genetic-algorithm-impl

Genetic algorithm pipeline is implemented on pictures which are in **/images** directory. 

Genetic algorithm finds the image that is similar to given. It starts from the lower left corner of the given size matrix and the point can only move in 8 directions. Every action has a cost. The low cost and high similarity route is preferred. Each step can be observed when the code is run, thanks to OpenCV. Different population sizes and mutation rates are tried to find best similarity. Results are shown in **/results** directory for each image. The files are recorded with the:

- population size
- mutation rate
- similarity rate they use, respectively. 

Comment out if change of parameters are requested on mutation or population or others.

Example run:

```
python3 genetic_impl.py -f [image file path] -d [dimension of working matrix] 
```
