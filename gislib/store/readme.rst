use indices when adding
implement retrieving
implement slicing instead of reprojection if transformation indicates roughly same.

Bottlenecks:
    - Reprojecting
    - Storing!


    Time stuff: Create a numpy array the shape of the source. Warp dataset into it. Selectively save parts of the data to locations.

- Determine location span
- Initalize numpy array for the hole span
- Load / create per location
- Create a dataset from a view of this array, according to the indices
- Reproject
- Save per location
