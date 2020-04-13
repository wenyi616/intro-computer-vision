
Assignment 3


PART A

> python3 photometric_stereo.py cat

> python3 photometric_stereo.py tentacle


PART A RESULTS were saved to the folder ./output/

--- 1.963299036026001 seconds ---

--- 8.154427766799927 seconds ---





PART B

> python3 plane_sweep_stereo.py tentacle

> python3 plane_sweep_stereo.py Flowers



PART B RESULTS were saved to the folder ./output/

Plane sweep took 20.72363305091858 seconds
Saving NCC to output/tentacle_ncc.png
Lossy conversion from int64 to uint8. Range [0, 254]. Convert image to uint8 prior to saving to suppress this warning.
Saving depth to output/tentacle_depth.npy
--- 29.03402280807495 seconds ---



Plane sweep took 17.645450830459595 seconds
Saving NCC to output/Flowers_ncc.png
Lossy conversion from int64 to uint8. Range [0, 254]. Convert image to uint8 prior to saving to suppress this warning.
Saving depth to output/Flowers_depth.npy
--- 23.253207206726074 seconds ---



