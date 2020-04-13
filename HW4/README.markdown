
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






OVERALL

Run the following test and passed all the tests in 24 seconds.
> nosetests -v

tests.anticorrelated_ncc_test ... ok
tests.compute_photometric_stereo_angle_test ... ok
tests.compute_photometric_stereo_full_test ... ok
tests.compute_photometric_stereo_half_albedo_test ... ok
tests.compute_photometric_stereo_test ... ok
tests.correlated_ncc_test ... ok
tests.ncc_full_identity_test ... ok
tests.ncc_full_offset_test ... ok
tests.ncc_full_shapes_test ... ok
tests.offset_and_scale_ncc_test ... ok
tests.offset_ncc_test ... ok
tests.preprocess_ncc_delta_test ... ok
tests.preprocess_ncc_full_test ... ok
tests.preprocess_ncc_uniform_test ... ok
tests.preprocess_ncc_zeros_test ... ok
tests.project_Rt_identity_20x10_test ... ok
tests.project_Rt_identity_centered_test ... ok
tests.project_Rt_identity_upperleft_test ... ok
tests.project_Rt_identity_xoff_test ... ok
tests.project_Rt_identity_yoff_test ... ok
tests.project_Rt_rot180_upperleft_test ... ok
tests.project_Rt_rot90_upperleft_test ... ok
tests.project_unproject_Rt_identity_randdepth_test ... ok
tests.project_unproject_Rt_identity_test ... ok
tests.project_unproject_Rt_random_randdepth_test ... ok
tests.scale_ncc_test ... ok
tests.zero_ncc_test ... ok

----------------------------------------------------------------------
Ran 27 tests in 24.196s

OK
