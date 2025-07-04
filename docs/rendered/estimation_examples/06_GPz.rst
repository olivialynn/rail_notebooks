GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4897678e-01	 3.2196596e-01	-3.3932259e-01	 3.1453413e-01	[-3.2589804e-01]	 4.6559143e-01


.. parsed-literal::

       2	-2.7771615e-01	 3.1158206e-01	-2.5394259e-01	 3.0418139e-01	[-2.3217974e-01]	 2.3399472e-01


.. parsed-literal::

       3	-2.3271557e-01	 2.9041925e-01	-1.9048008e-01	 2.8130729e-01	[-1.5574457e-01]	 2.9580259e-01
       4	-1.9744351e-01	 2.6713240e-01	-1.5686219e-01	 2.5902426e-01	[-1.1235265e-01]	 1.9863534e-01


.. parsed-literal::

       5	-1.0468093e-01	 2.5752143e-01	-7.0257052e-02	 2.5141549e-01	[-4.3853602e-02]	 2.1301770e-01


.. parsed-literal::

       6	-7.5727265e-02	 2.5346317e-01	-4.6114354e-02	 2.4840970e-01	[-2.7975085e-02]	 2.0976210e-01
       7	-5.6136792e-02	 2.4995092e-01	-3.2627953e-02	 2.4507679e-01	[-1.3542991e-02]	 1.7663121e-01


.. parsed-literal::

       8	-4.5336375e-02	 2.4815932e-01	-2.5253176e-02	 2.4299305e-01	[-4.6885937e-03]	 2.1739459e-01


.. parsed-literal::

       9	-3.3266909e-02	 2.4595017e-01	-1.5853599e-02	 2.4080334e-01	[ 4.5679257e-03]	 2.0774198e-01


.. parsed-literal::

      10	-2.1279845e-02	 2.4347605e-01	-5.6448813e-03	 2.3898357e-01	[ 1.3307914e-02]	 2.1734953e-01


.. parsed-literal::

      11	-1.6801570e-02	 2.4267580e-01	-2.0422558e-03	 2.3811701e-01	[ 1.5440313e-02]	 3.2060504e-01


.. parsed-literal::

      12	-1.3104243e-02	 2.4214852e-01	 1.1223292e-03	 2.3750463e-01	[ 1.9736319e-02]	 2.1184564e-01
      13	-9.1575765e-03	 2.4145249e-01	 4.7436288e-03	 2.3646805e-01	[ 2.4896373e-02]	 1.7697382e-01


.. parsed-literal::

      14	-2.1320168e-03	 2.4008146e-01	 1.2551573e-02	 2.3438893e-01	[ 3.6281253e-02]	 2.1649861e-01


.. parsed-literal::

      15	 1.4378508e-01	 2.2439129e-01	 1.6668808e-01	 2.1791625e-01	[ 1.7999117e-01]	 3.2045507e-01


.. parsed-literal::

      16	 1.6948244e-01	 2.2119307e-01	 1.9165266e-01	 2.1518049e-01	[ 2.0049188e-01]	 3.3595562e-01


.. parsed-literal::

      17	 2.3388337e-01	 2.1869136e-01	 2.5890654e-01	 2.1337298e-01	[ 2.6643509e-01]	 2.1272898e-01


.. parsed-literal::

      18	 2.8991127e-01	 2.2071530e-01	 3.2242883e-01	 2.1552413e-01	[ 3.2259281e-01]	 2.1129799e-01


.. parsed-literal::

      19	 3.3418028e-01	 2.1852117e-01	 3.6738079e-01	 2.1220065e-01	[ 3.6577828e-01]	 2.0570779e-01
      20	 3.8064135e-01	 2.1340246e-01	 4.1480943e-01	 2.0930915e-01	[ 4.0338190e-01]	 1.9702840e-01


.. parsed-literal::

      21	 4.1965222e-01	 2.1016847e-01	 4.5302741e-01	 2.0704081e-01	[ 4.3535663e-01]	 1.9882727e-01


.. parsed-literal::

      22	 4.7934743e-01	 2.0718002e-01	 5.1300856e-01	 2.0387680e-01	[ 4.9099030e-01]	 2.0893645e-01


.. parsed-literal::

      23	 5.5824924e-01	 2.0402810e-01	 5.9460213e-01	 2.0078900e-01	[ 5.5898187e-01]	 2.0946813e-01
      24	 6.0085966e-01	 2.0455781e-01	 6.4095380e-01	 2.0030210e-01	[ 5.9546752e-01]	 1.7136264e-01


.. parsed-literal::

      25	 6.4124978e-01	 1.9977658e-01	 6.7997566e-01	 1.9671509e-01	[ 6.4387069e-01]	 2.0942354e-01


.. parsed-literal::

      26	 6.6344996e-01	 1.9864889e-01	 7.0185550e-01	 1.9571365e-01	[ 6.6841164e-01]	 2.0627308e-01


.. parsed-literal::

      27	 7.0104796e-01	 1.9967956e-01	 7.3810504e-01	 1.9707367e-01	[ 7.0709096e-01]	 2.1040535e-01


.. parsed-literal::

      28	 7.4419813e-01	 1.9815134e-01	 7.8096100e-01	 1.9379453e-01	[ 7.5993172e-01]	 2.0780849e-01


.. parsed-literal::

      29	 7.7713665e-01	 1.9768923e-01	 8.1651472e-01	 1.9226526e-01	[ 7.8863211e-01]	 2.0864391e-01


.. parsed-literal::

      30	 8.0124126e-01	 1.9946084e-01	 8.4020878e-01	 1.9248564e-01	[ 8.1323310e-01]	 2.1025968e-01


.. parsed-literal::

      31	 8.1670607e-01	 1.9516834e-01	 8.5575595e-01	 1.8870590e-01	[ 8.3176646e-01]	 2.0475984e-01


.. parsed-literal::

      32	 8.3518697e-01	 1.9735411e-01	 8.7376928e-01	 1.9048858e-01	[ 8.5442091e-01]	 2.0880556e-01
      33	 8.6325832e-01	 1.9545351e-01	 9.0359001e-01	 1.8859769e-01	[ 8.7783109e-01]	 1.8410897e-01


.. parsed-literal::

      34	 8.8286067e-01	 1.9538801e-01	 9.2426069e-01	 1.8882117e-01	[ 8.9417585e-01]	 2.1114612e-01


.. parsed-literal::

      35	 8.9986284e-01	 1.9445970e-01	 9.4193732e-01	 1.8798797e-01	[ 9.1127143e-01]	 2.0770931e-01


.. parsed-literal::

      36	 9.2357068e-01	 1.9245124e-01	 9.6594764e-01	 1.8630535e-01	[ 9.3414617e-01]	 2.1432829e-01


.. parsed-literal::

      37	 9.3225560e-01	 1.9214336e-01	 9.7601051e-01	 1.8590493e-01	[ 9.3604084e-01]	 2.1330118e-01


.. parsed-literal::

      38	 9.5726246e-01	 1.8977435e-01	 1.0002602e+00	 1.8355105e-01	[ 9.6352645e-01]	 2.0997810e-01


.. parsed-literal::

      39	 9.6795164e-01	 1.8857467e-01	 1.0109565e+00	 1.8216371e-01	[ 9.7246873e-01]	 2.1062756e-01
      40	 9.8754692e-01	 1.8584596e-01	 1.0309921e+00	 1.7921311e-01	[ 9.8497843e-01]	 1.9842482e-01


.. parsed-literal::

      41	 1.0099057e+00	 1.8230611e-01	 1.0548043e+00	 1.7573396e-01	[ 9.9530509e-01]	 2.1666980e-01


.. parsed-literal::

      42	 1.0210476e+00	 1.7920322e-01	 1.0668146e+00	 1.7282146e-01	[ 9.9695283e-01]	 2.0228601e-01
      43	 1.0332557e+00	 1.7878436e-01	 1.0788584e+00	 1.7234789e-01	[ 1.0155685e+00]	 1.7678022e-01


.. parsed-literal::

      44	 1.0410895e+00	 1.7784775e-01	 1.0870776e+00	 1.7151841e-01	[ 1.0238559e+00]	 1.7684364e-01


.. parsed-literal::

      45	 1.0543730e+00	 1.7638812e-01	 1.1008992e+00	 1.6974265e-01	[ 1.0369814e+00]	 2.1853495e-01
      46	 1.0660066e+00	 1.7399658e-01	 1.1129928e+00	 1.6732707e-01	[ 1.0524950e+00]	 1.8312407e-01


.. parsed-literal::

      47	 1.0790593e+00	 1.7261508e-01	 1.1258329e+00	 1.6549403e-01	[ 1.0639192e+00]	 2.0519280e-01
      48	 1.0892064e+00	 1.7124460e-01	 1.1355842e+00	 1.6379511e-01	[ 1.0709222e+00]	 1.8788290e-01


.. parsed-literal::

      49	 1.0988678e+00	 1.6976393e-01	 1.1454458e+00	 1.6171984e-01	[ 1.0794997e+00]	 1.8533182e-01


.. parsed-literal::

      50	 1.1094573e+00	 1.6699901e-01	 1.1559025e+00	 1.5855888e-01	[ 1.0906411e+00]	 2.2278047e-01


.. parsed-literal::

      51	 1.1213895e+00	 1.6541553e-01	 1.1678548e+00	 1.5714075e-01	[ 1.1019049e+00]	 2.0222974e-01


.. parsed-literal::

      52	 1.1322455e+00	 1.6412668e-01	 1.1791162e+00	 1.5585072e-01	[ 1.1128014e+00]	 2.0955491e-01


.. parsed-literal::

      53	 1.1486991e+00	 1.6145487e-01	 1.1960999e+00	 1.5387847e-01	[ 1.1262673e+00]	 2.1644115e-01
      54	 1.1575015e+00	 1.6042763e-01	 1.2062871e+00	 1.5270026e-01	[ 1.1443262e+00]	 2.0181847e-01


.. parsed-literal::

      55	 1.1724323e+00	 1.5850389e-01	 1.2205689e+00	 1.5114243e-01	[ 1.1562136e+00]	 2.0173454e-01


.. parsed-literal::

      56	 1.1792893e+00	 1.5766979e-01	 1.2274167e+00	 1.5017271e-01	[ 1.1625538e+00]	 2.0949674e-01


.. parsed-literal::

      57	 1.1892765e+00	 1.5767231e-01	 1.2380741e+00	 1.4948918e-01	[ 1.1693612e+00]	 2.1895289e-01
      58	 1.2011470e+00	 1.5627278e-01	 1.2501827e+00	 1.4760622e-01	[ 1.1844623e+00]	 1.8077779e-01


.. parsed-literal::

      59	 1.2081429e+00	 1.5534891e-01	 1.2573667e+00	 1.4659147e-01	[ 1.1924913e+00]	 2.1237159e-01
      60	 1.2213951e+00	 1.5430088e-01	 1.2715106e+00	 1.4584746e-01	[ 1.2019508e+00]	 1.9127893e-01


.. parsed-literal::

      61	 1.2305564e+00	 1.5252002e-01	 1.2808329e+00	 1.4406318e-01	[ 1.2051227e+00]	 1.7356849e-01


.. parsed-literal::

      62	 1.2400257e+00	 1.5224124e-01	 1.2902537e+00	 1.4410260e-01	[ 1.2083999e+00]	 2.1270776e-01


.. parsed-literal::

      63	 1.2492586e+00	 1.5136288e-01	 1.2996650e+00	 1.4356745e-01	[ 1.2091163e+00]	 2.0739412e-01


.. parsed-literal::

      64	 1.2582090e+00	 1.5046915e-01	 1.3085385e+00	 1.4278611e-01	[ 1.2112193e+00]	 2.1239901e-01


.. parsed-literal::

      65	 1.2676763e+00	 1.4967913e-01	 1.3179553e+00	 1.4194766e-01	[ 1.2175826e+00]	 2.1200418e-01


.. parsed-literal::

      66	 1.2750793e+00	 1.4964190e-01	 1.3254647e+00	 1.4203599e-01	[ 1.2203973e+00]	 2.0944333e-01


.. parsed-literal::

      67	 1.2842451e+00	 1.4843822e-01	 1.3345234e+00	 1.4118502e-01	[ 1.2295060e+00]	 2.1590972e-01
      68	 1.2905824e+00	 1.4739202e-01	 1.3411175e+00	 1.4048608e-01	[ 1.2297693e+00]	 1.7530131e-01


.. parsed-literal::

      69	 1.2993013e+00	 1.4594927e-01	 1.3502545e+00	 1.3974567e-01	  1.2273351e+00 	 2.0744371e-01


.. parsed-literal::

      70	 1.3043791e+00	 1.4501754e-01	 1.3556557e+00	 1.3886007e-01	  1.2260068e+00 	 3.0531192e-01


.. parsed-literal::

      71	 1.3111472e+00	 1.4463553e-01	 1.3625393e+00	 1.3863232e-01	  1.2264037e+00 	 2.1101475e-01
      72	 1.3202351e+00	 1.4388872e-01	 1.3717835e+00	 1.3770088e-01	  1.2296992e+00 	 1.9825077e-01


.. parsed-literal::

      73	 1.3305019e+00	 1.4390228e-01	 1.3820765e+00	 1.3695572e-01	[ 1.2360794e+00]	 2.0758843e-01
      74	 1.3373955e+00	 1.4215377e-01	 1.3889079e+00	 1.3525230e-01	[ 1.2475807e+00]	 1.9187498e-01


.. parsed-literal::

      75	 1.3423364e+00	 1.4175280e-01	 1.3936821e+00	 1.3485302e-01	[ 1.2546204e+00]	 2.1124792e-01


.. parsed-literal::

      76	 1.3485017e+00	 1.4103443e-01	 1.4000136e+00	 1.3395463e-01	[ 1.2567411e+00]	 2.0946980e-01


.. parsed-literal::

      77	 1.3544470e+00	 1.4044470e-01	 1.4061432e+00	 1.3318327e-01	[ 1.2593989e+00]	 2.1412277e-01


.. parsed-literal::

      78	 1.3579714e+00	 1.3912856e-01	 1.4102633e+00	 1.3127601e-01	  1.2572824e+00 	 2.1043849e-01


.. parsed-literal::

      79	 1.3659599e+00	 1.3894237e-01	 1.4179510e+00	 1.3130440e-01	[ 1.2636330e+00]	 2.1023679e-01


.. parsed-literal::

      80	 1.3691473e+00	 1.3882959e-01	 1.4210874e+00	 1.3119408e-01	[ 1.2667384e+00]	 2.1240973e-01


.. parsed-literal::

      81	 1.3750211e+00	 1.3830453e-01	 1.4271373e+00	 1.3055005e-01	[ 1.2681416e+00]	 2.1590137e-01


.. parsed-literal::

      82	 1.3776027e+00	 1.3836946e-01	 1.4299619e+00	 1.3036078e-01	  1.2663120e+00 	 2.0708251e-01
      83	 1.3834771e+00	 1.3787361e-01	 1.4357056e+00	 1.2988204e-01	[ 1.2710918e+00]	 1.7254305e-01


.. parsed-literal::

      84	 1.3864941e+00	 1.3754319e-01	 1.4387116e+00	 1.2957990e-01	[ 1.2721540e+00]	 2.1841550e-01


.. parsed-literal::

      85	 1.3900983e+00	 1.3715330e-01	 1.4423228e+00	 1.2926562e-01	[ 1.2724133e+00]	 2.1300673e-01


.. parsed-literal::

      86	 1.3951270e+00	 1.3676761e-01	 1.4475327e+00	 1.2908202e-01	  1.2667368e+00 	 2.0364857e-01


.. parsed-literal::

      87	 1.3993563e+00	 1.3609970e-01	 1.4518002e+00	 1.2885168e-01	  1.2701530e+00 	 2.1817279e-01


.. parsed-literal::

      88	 1.4025704e+00	 1.3609473e-01	 1.4549136e+00	 1.2877499e-01	[ 1.2740560e+00]	 2.0957780e-01


.. parsed-literal::

      89	 1.4060670e+00	 1.3586597e-01	 1.4584356e+00	 1.2860847e-01	[ 1.2776469e+00]	 2.0131087e-01
      90	 1.4098403e+00	 1.3556328e-01	 1.4622856e+00	 1.2843103e-01	[ 1.2805090e+00]	 1.7605734e-01


.. parsed-literal::

      91	 1.4151868e+00	 1.3455128e-01	 1.4678227e+00	 1.2773635e-01	[ 1.2847861e+00]	 2.1837759e-01
      92	 1.4198853e+00	 1.3399421e-01	 1.4725736e+00	 1.2737761e-01	[ 1.2894208e+00]	 1.8212199e-01


.. parsed-literal::

      93	 1.4241425e+00	 1.3351997e-01	 1.4768693e+00	 1.2696214e-01	[ 1.2912259e+00]	 2.1605420e-01


.. parsed-literal::

      94	 1.4282569e+00	 1.3316593e-01	 1.4811228e+00	 1.2673989e-01	  1.2907777e+00 	 2.1829247e-01


.. parsed-literal::

      95	 1.4324501e+00	 1.3288532e-01	 1.4855657e+00	 1.2627280e-01	  1.2867417e+00 	 2.1795106e-01
      96	 1.4355519e+00	 1.3254230e-01	 1.4887081e+00	 1.2634367e-01	  1.2895421e+00 	 2.0779395e-01


.. parsed-literal::

      97	 1.4382722e+00	 1.3247635e-01	 1.4913715e+00	 1.2634581e-01	[ 1.2929000e+00]	 2.1172690e-01


.. parsed-literal::

      98	 1.4411815e+00	 1.3221961e-01	 1.4943284e+00	 1.2622083e-01	  1.2924155e+00 	 2.2193503e-01


.. parsed-literal::

      99	 1.4437004e+00	 1.3213122e-01	 1.4969330e+00	 1.2637261e-01	  1.2925955e+00 	 2.1726823e-01


.. parsed-literal::

     100	 1.4466212e+00	 1.3187742e-01	 1.4999621e+00	 1.2630378e-01	  1.2888047e+00 	 2.1423745e-01


.. parsed-literal::

     101	 1.4507381e+00	 1.3152729e-01	 1.5042007e+00	 1.2617685e-01	  1.2845120e+00 	 2.1289134e-01


.. parsed-literal::

     102	 1.4535493e+00	 1.3116557e-01	 1.5071038e+00	 1.2625193e-01	  1.2741485e+00 	 2.1679926e-01


.. parsed-literal::

     103	 1.4568024e+00	 1.3101143e-01	 1.5102780e+00	 1.2604852e-01	  1.2777524e+00 	 2.0816875e-01


.. parsed-literal::

     104	 1.4599663e+00	 1.3067997e-01	 1.5134799e+00	 1.2578280e-01	  1.2777856e+00 	 2.1791601e-01
     105	 1.4622524e+00	 1.3061982e-01	 1.5157877e+00	 1.2583250e-01	  1.2760915e+00 	 1.8663144e-01


.. parsed-literal::

     106	 1.4647391e+00	 1.3041547e-01	 1.5183136e+00	 1.2586075e-01	  1.2718217e+00 	 2.1093297e-01
     107	 1.4676758e+00	 1.3023486e-01	 1.5213241e+00	 1.2594036e-01	  1.2650324e+00 	 1.9973803e-01


.. parsed-literal::

     108	 1.4699498e+00	 1.3017157e-01	 1.5236298e+00	 1.2587582e-01	  1.2631916e+00 	 2.0855856e-01


.. parsed-literal::

     109	 1.4727444e+00	 1.3021712e-01	 1.5265404e+00	 1.2578298e-01	  1.2598979e+00 	 2.0655584e-01


.. parsed-literal::

     110	 1.4757278e+00	 1.3030870e-01	 1.5294698e+00	 1.2562438e-01	  1.2570730e+00 	 2.1494913e-01


.. parsed-literal::

     111	 1.4773882e+00	 1.3021308e-01	 1.5310736e+00	 1.2548310e-01	  1.2618274e+00 	 2.1553779e-01


.. parsed-literal::

     112	 1.4799826e+00	 1.3010694e-01	 1.5336526e+00	 1.2525032e-01	  1.2660932e+00 	 2.0965195e-01


.. parsed-literal::

     113	 1.4816729e+00	 1.2991822e-01	 1.5354133e+00	 1.2493291e-01	  1.2647189e+00 	 2.0233274e-01
     114	 1.4834642e+00	 1.2986676e-01	 1.5371894e+00	 1.2493444e-01	  1.2643402e+00 	 1.7513418e-01


.. parsed-literal::

     115	 1.4853214e+00	 1.2977099e-01	 1.5390811e+00	 1.2491216e-01	  1.2617001e+00 	 1.8281078e-01


.. parsed-literal::

     116	 1.4869295e+00	 1.2971760e-01	 1.5407177e+00	 1.2477913e-01	  1.2627551e+00 	 2.0103860e-01


.. parsed-literal::

     117	 1.4909602e+00	 1.2964998e-01	 1.5449191e+00	 1.2443988e-01	  1.2645848e+00 	 2.1195292e-01


.. parsed-literal::

     118	 1.4929472e+00	 1.2959922e-01	 1.5469817e+00	 1.2414001e-01	  1.2705212e+00 	 3.1957364e-01


.. parsed-literal::

     119	 1.4948988e+00	 1.2953908e-01	 1.5489590e+00	 1.2403376e-01	  1.2715930e+00 	 2.0489717e-01
     120	 1.4966486e+00	 1.2953026e-01	 1.5507419e+00	 1.2399887e-01	  1.2711552e+00 	 1.9994974e-01


.. parsed-literal::

     121	 1.4984367e+00	 1.2943944e-01	 1.5525783e+00	 1.2400318e-01	  1.2679826e+00 	 2.1493030e-01


.. parsed-literal::

     122	 1.5010824e+00	 1.2925972e-01	 1.5554511e+00	 1.2386827e-01	  1.2537546e+00 	 2.0841193e-01


.. parsed-literal::

     123	 1.5032976e+00	 1.2913548e-01	 1.5576674e+00	 1.2374170e-01	  1.2533919e+00 	 2.1064448e-01


.. parsed-literal::

     124	 1.5045844e+00	 1.2906533e-01	 1.5589322e+00	 1.2355732e-01	  1.2532143e+00 	 2.0688105e-01


.. parsed-literal::

     125	 1.5069218e+00	 1.2887646e-01	 1.5612963e+00	 1.2307364e-01	  1.2507443e+00 	 2.0587349e-01


.. parsed-literal::

     126	 1.5083705e+00	 1.2882662e-01	 1.5627680e+00	 1.2267130e-01	  1.2508493e+00 	 3.1419253e-01


.. parsed-literal::

     127	 1.5105483e+00	 1.2856554e-01	 1.5649550e+00	 1.2225234e-01	  1.2489533e+00 	 2.3817015e-01


.. parsed-literal::

     128	 1.5122962e+00	 1.2835177e-01	 1.5666902e+00	 1.2193361e-01	  1.2486119e+00 	 2.0412207e-01


.. parsed-literal::

     129	 1.5139085e+00	 1.2810636e-01	 1.5682780e+00	 1.2177125e-01	  1.2500887e+00 	 2.0766330e-01
     130	 1.5155757e+00	 1.2798433e-01	 1.5698896e+00	 1.2170808e-01	  1.2495300e+00 	 1.8202734e-01


.. parsed-literal::

     131	 1.5173321e+00	 1.2788490e-01	 1.5716030e+00	 1.2166305e-01	  1.2528212e+00 	 1.9048619e-01


.. parsed-literal::

     132	 1.5184617e+00	 1.2779844e-01	 1.5727663e+00	 1.2172841e-01	  1.2461720e+00 	 2.1086478e-01
     133	 1.5199394e+00	 1.2779935e-01	 1.5742123e+00	 1.2159964e-01	  1.2480468e+00 	 1.7527843e-01


.. parsed-literal::

     134	 1.5205822e+00	 1.2780316e-01	 1.5748580e+00	 1.2150013e-01	  1.2473171e+00 	 2.1163011e-01
     135	 1.5220384e+00	 1.2781003e-01	 1.5763459e+00	 1.2130743e-01	  1.2463889e+00 	 1.9925213e-01


.. parsed-literal::

     136	 1.5240420e+00	 1.2778660e-01	 1.5783976e+00	 1.2104917e-01	  1.2461072e+00 	 2.0755696e-01


.. parsed-literal::

     137	 1.5251615e+00	 1.2782329e-01	 1.5797483e+00	 1.2084427e-01	  1.2494600e+00 	 2.1802068e-01


.. parsed-literal::

     138	 1.5277240e+00	 1.2772064e-01	 1.5821616e+00	 1.2071277e-01	  1.2524913e+00 	 2.1556735e-01
     139	 1.5287774e+00	 1.2762060e-01	 1.5831740e+00	 1.2070792e-01	  1.2540214e+00 	 1.9864750e-01


.. parsed-literal::

     140	 1.5301542e+00	 1.2760376e-01	 1.5845572e+00	 1.2060559e-01	  1.2545818e+00 	 2.1090484e-01
     141	 1.5314749e+00	 1.2744394e-01	 1.5859075e+00	 1.2048866e-01	  1.2555144e+00 	 1.9600058e-01


.. parsed-literal::

     142	 1.5326870e+00	 1.2744552e-01	 1.5871274e+00	 1.2034655e-01	  1.2532161e+00 	 2.0891237e-01
     143	 1.5347034e+00	 1.2747388e-01	 1.5892349e+00	 1.1993335e-01	  1.2485885e+00 	 1.8634295e-01


.. parsed-literal::

     144	 1.5356994e+00	 1.2748406e-01	 1.5902948e+00	 1.1983660e-01	  1.2400702e+00 	 2.0157242e-01
     145	 1.5368171e+00	 1.2746866e-01	 1.5914113e+00	 1.1971974e-01	  1.2398672e+00 	 1.8525386e-01


.. parsed-literal::

     146	 1.5379709e+00	 1.2739494e-01	 1.5925909e+00	 1.1964119e-01	  1.2404877e+00 	 2.0320678e-01
     147	 1.5389912e+00	 1.2729527e-01	 1.5936274e+00	 1.1963479e-01	  1.2371499e+00 	 1.8725395e-01


.. parsed-literal::

     148	 1.5401282e+00	 1.2699444e-01	 1.5948986e+00	 1.1961327e-01	  1.2297513e+00 	 1.9892573e-01
     149	 1.5416686e+00	 1.2693347e-01	 1.5963635e+00	 1.1969715e-01	  1.2215334e+00 	 1.7488146e-01


.. parsed-literal::

     150	 1.5422385e+00	 1.2692353e-01	 1.5969085e+00	 1.1969523e-01	  1.2203876e+00 	 2.1016860e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.13 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff40ce3c970>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.79 s, sys: 48.9 ms, total: 1.84 s
    Wall time: 614 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

