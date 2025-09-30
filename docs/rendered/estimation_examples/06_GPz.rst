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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3561314e-01	 3.1839663e-01	-3.2587270e-01	 3.2938263e-01	[-3.4643409e-01]	 4.6058750e-01


.. parsed-literal::

       2	-2.6452204e-01	 3.0718848e-01	-2.3996295e-01	 3.1894443e-01	[-2.7289978e-01]	 2.2743464e-01


.. parsed-literal::

       3	-2.1928754e-01	 2.8614482e-01	-1.7566897e-01	 2.9846299e-01	[-2.2347463e-01]	 2.8216124e-01
       4	-1.9405744e-01	 2.6214870e-01	-1.5279344e-01	 2.7560804e-01	 -2.3527138e-01 	 1.9592929e-01


.. parsed-literal::

       5	-9.7425089e-02	 2.5487350e-01	-6.1057792e-02	 2.7246247e-01	[-1.3285343e-01]	 2.0905471e-01


.. parsed-literal::

       6	-6.0717148e-02	 2.4836314e-01	-2.7507475e-02	 2.6185766e-01	[-7.7983575e-02]	 2.1025229e-01


.. parsed-literal::

       7	-4.1729441e-02	 2.4582468e-01	-1.6493897e-02	 2.6134979e-01	[-7.6292618e-02]	 2.0194364e-01


.. parsed-literal::

       8	-2.7778654e-02	 2.4354830e-01	-6.8522411e-03	 2.6057563e-01	[-7.3301793e-02]	 2.1090913e-01
       9	-1.2921259e-02	 2.4089019e-01	 4.6289722e-03	 2.5795455e-01	[-6.4381081e-02]	 1.9940495e-01


.. parsed-literal::

      10	-5.4861633e-03	 2.3989911e-01	 9.7120248e-03	 2.5531076e-01	[-5.2974153e-02]	 2.1428633e-01


.. parsed-literal::

      11	 9.2046119e-04	 2.3849402e-01	 1.5525809e-02	 2.5374723e-01	[-4.6586840e-02]	 2.0688510e-01


.. parsed-literal::

      12	 3.2655446e-03	 2.3804011e-01	 1.7603359e-02	 2.5364821e-01	[-4.5972718e-02]	 2.0718336e-01


.. parsed-literal::

      13	 7.8963403e-03	 2.3713502e-01	 2.2045775e-02	 2.5317368e-01	[-4.1618068e-02]	 2.1262503e-01
      14	 1.8112173e-02	 2.3480785e-01	 3.4355868e-02	 2.5092360e-01	[-3.2541305e-02]	 1.9049120e-01


.. parsed-literal::

      15	 1.0833903e-01	 2.2235184e-01	 1.3589647e-01	 2.4625566e-01	[ 4.5556181e-02]	 2.0850611e-01
      16	 2.2086657e-01	 2.2323942e-01	 2.5289838e-01	 2.3690830e-01	[ 2.0639225e-01]	 1.8733478e-01


.. parsed-literal::

      17	 2.5895249e-01	 2.1445669e-01	 2.8905566e-01	 2.2690457e-01	[ 2.4630563e-01]	 1.9249535e-01
      18	 2.9835851e-01	 2.0925976e-01	 3.2962387e-01	 2.2295743e-01	[ 2.7992236e-01]	 1.7443466e-01


.. parsed-literal::

      19	 3.4792230e-01	 2.0458932e-01	 3.8371787e-01	 2.2155190e-01	[ 3.1103494e-01]	 1.9657111e-01
      20	 3.9990349e-01	 2.0217320e-01	 4.3369472e-01	 2.1998392e-01	[ 3.5392841e-01]	 1.7435551e-01


.. parsed-literal::

      21	 4.4146734e-01	 1.9972253e-01	 4.7485693e-01	 2.1808370e-01	[ 4.0186222e-01]	 2.0762730e-01


.. parsed-literal::

      22	 5.2557780e-01	 1.9739131e-01	 5.6057925e-01	 2.1497611e-01	[ 4.8721815e-01]	 2.0563126e-01


.. parsed-literal::

      23	 5.6228635e-01	 2.0202596e-01	 6.0417293e-01	 2.1501953e-01	[ 5.0828184e-01]	 2.1908188e-01


.. parsed-literal::

      24	 6.4179196e-01	 1.9616009e-01	 6.7935017e-01	 2.0882424e-01	[ 6.1436439e-01]	 2.1028781e-01


.. parsed-literal::

      25	 6.6834347e-01	 1.9263019e-01	 7.0578043e-01	 2.0528624e-01	[ 6.4096043e-01]	 2.1527100e-01
      26	 7.0286526e-01	 1.9034641e-01	 7.4096734e-01	 2.0283583e-01	[ 6.6570953e-01]	 1.9665694e-01


.. parsed-literal::

      27	 7.4592362e-01	 1.8943467e-01	 7.8464660e-01	 2.0526485e-01	[ 6.9670934e-01]	 2.1492124e-01


.. parsed-literal::

      28	 7.9114072e-01	 1.9807514e-01	 8.3047552e-01	 2.1836233e-01	[ 7.2849453e-01]	 2.0614457e-01


.. parsed-literal::

      29	 8.2702585e-01	 1.9916249e-01	 8.6606855e-01	 2.1987455e-01	[ 7.8398284e-01]	 2.1039557e-01


.. parsed-literal::

      30	 8.5614835e-01	 1.9317306e-01	 8.9660314e-01	 2.1322366e-01	[ 8.1144227e-01]	 2.1512055e-01


.. parsed-literal::

      31	 8.8393333e-01	 1.8964187e-01	 9.2597728e-01	 2.1001032e-01	[ 8.4234158e-01]	 2.0785689e-01


.. parsed-literal::

      32	 9.0265334e-01	 1.8325983e-01	 9.4555835e-01	 2.0346144e-01	[ 8.5742836e-01]	 2.1600437e-01


.. parsed-literal::

      33	 9.2035046e-01	 1.8125934e-01	 9.6284719e-01	 2.0079518e-01	[ 8.7535523e-01]	 2.1347737e-01


.. parsed-literal::

      34	 9.3961069e-01	 1.7933442e-01	 9.8183019e-01	 1.9884517e-01	[ 8.9284416e-01]	 2.0988059e-01


.. parsed-literal::

      35	 9.5478765e-01	 1.7586157e-01	 9.9705425e-01	 1.9502043e-01	[ 9.0581587e-01]	 2.0717430e-01
      36	 9.7302865e-01	 1.7293704e-01	 1.0157029e+00	 1.9186505e-01	[ 9.2155051e-01]	 1.9901729e-01


.. parsed-literal::

      37	 9.9132961e-01	 1.6964556e-01	 1.0345803e+00	 1.8751853e-01	[ 9.4333081e-01]	 2.0700502e-01
      38	 1.0096374e+00	 1.6549727e-01	 1.0535367e+00	 1.8183693e-01	[ 9.6571064e-01]	 1.9665146e-01


.. parsed-literal::

      39	 1.0254591e+00	 1.6215047e-01	 1.0699701e+00	 1.7745891e-01	[ 9.8411625e-01]	 2.1577787e-01


.. parsed-literal::

      40	 1.0504672e+00	 1.5745963e-01	 1.0967941e+00	 1.7149488e-01	[ 1.0129410e+00]	 2.0194077e-01
      41	 1.0524376e+00	 1.5665643e-01	 1.0999593e+00	 1.6977042e-01	[ 1.0296191e+00]	 1.8183374e-01


.. parsed-literal::

      42	 1.0692134e+00	 1.5596099e-01	 1.1154632e+00	 1.6872639e-01	[ 1.0443556e+00]	 2.0255637e-01


.. parsed-literal::

      43	 1.0783032e+00	 1.5433796e-01	 1.1246386e+00	 1.6713067e-01	[ 1.0481283e+00]	 2.0360351e-01
      44	 1.0871761e+00	 1.5328263e-01	 1.1335975e+00	 1.6619781e-01	[ 1.0557472e+00]	 1.8284678e-01


.. parsed-literal::

      45	 1.1099777e+00	 1.4971551e-01	 1.1571540e+00	 1.6406000e-01	[ 1.0685521e+00]	 2.0823479e-01
      46	 1.1130144e+00	 1.4740665e-01	 1.1608353e+00	 1.6184661e-01	[ 1.0750215e+00]	 1.8403578e-01


.. parsed-literal::

      47	 1.1306992e+00	 1.4614733e-01	 1.1780092e+00	 1.6067707e-01	[ 1.0914777e+00]	 1.9717598e-01
      48	 1.1378727e+00	 1.4568097e-01	 1.1851602e+00	 1.6067848e-01	[ 1.0966333e+00]	 1.9831729e-01


.. parsed-literal::

      49	 1.1451665e+00	 1.4484690e-01	 1.1926218e+00	 1.6065946e-01	[ 1.1016764e+00]	 2.0942044e-01
      50	 1.1595890e+00	 1.4330146e-01	 1.2071837e+00	 1.5961251e-01	[ 1.1134086e+00]	 1.7268443e-01


.. parsed-literal::

      51	 1.1711229e+00	 1.4176285e-01	 1.2188096e+00	 1.5922053e-01	[ 1.1179645e+00]	 1.8195605e-01
      52	 1.1793322e+00	 1.4018622e-01	 1.2270757e+00	 1.5741867e-01	[ 1.1252491e+00]	 1.7983913e-01


.. parsed-literal::

      53	 1.1944129e+00	 1.3678837e-01	 1.2424733e+00	 1.5241310e-01	[ 1.1356159e+00]	 1.9410944e-01
      54	 1.2047705e+00	 1.3485566e-01	 1.2524583e+00	 1.5067475e-01	[ 1.1494962e+00]	 1.7464542e-01


.. parsed-literal::

      55	 1.2128988e+00	 1.3421377e-01	 1.2604321e+00	 1.4976994e-01	[ 1.1606546e+00]	 2.0600200e-01
      56	 1.2248920e+00	 1.3328441e-01	 1.2727668e+00	 1.4863676e-01	[ 1.1735016e+00]	 2.0006800e-01


.. parsed-literal::

      57	 1.2332135e+00	 1.3195711e-01	 1.2814229e+00	 1.4797092e-01	[ 1.1816236e+00]	 2.1202850e-01


.. parsed-literal::

      58	 1.2426539e+00	 1.3121315e-01	 1.2910973e+00	 1.4718656e-01	[ 1.1875068e+00]	 2.1078634e-01
      59	 1.2545462e+00	 1.3036494e-01	 1.3032972e+00	 1.4758583e-01	[ 1.1910747e+00]	 1.8226600e-01


.. parsed-literal::

      60	 1.2631997e+00	 1.2941943e-01	 1.3123508e+00	 1.4660594e-01	[ 1.1943671e+00]	 2.0769334e-01


.. parsed-literal::

      61	 1.2712014e+00	 1.2995602e-01	 1.3201582e+00	 1.4852222e-01	[ 1.2034663e+00]	 2.0430827e-01


.. parsed-literal::

      62	 1.2778521e+00	 1.2939711e-01	 1.3266906e+00	 1.4788731e-01	[ 1.2112654e+00]	 2.0874619e-01


.. parsed-literal::

      63	 1.2819899e+00	 1.2910559e-01	 1.3309218e+00	 1.4736524e-01	[ 1.2162674e+00]	 2.1444130e-01
      64	 1.2909148e+00	 1.2853596e-01	 1.3402649e+00	 1.4655118e-01	[ 1.2252113e+00]	 1.9957733e-01


.. parsed-literal::

      65	 1.2976849e+00	 1.2813035e-01	 1.3472953e+00	 1.4565454e-01	[ 1.2326825e+00]	 1.8876386e-01
      66	 1.3058750e+00	 1.2776306e-01	 1.3555155e+00	 1.4560983e-01	[ 1.2399336e+00]	 1.9224000e-01


.. parsed-literal::

      67	 1.3127106e+00	 1.2750851e-01	 1.3623016e+00	 1.4613371e-01	[ 1.2436401e+00]	 2.0546722e-01


.. parsed-literal::

      68	 1.3187799e+00	 1.2722316e-01	 1.3683306e+00	 1.4641244e-01	[ 1.2497110e+00]	 2.0198774e-01


.. parsed-literal::

      69	 1.3263220e+00	 1.2709268e-01	 1.3759935e+00	 1.4683735e-01	[ 1.2545322e+00]	 2.0906997e-01


.. parsed-literal::

      70	 1.3343250e+00	 1.2669849e-01	 1.3839503e+00	 1.4654632e-01	[ 1.2629614e+00]	 2.0591807e-01
      71	 1.3417226e+00	 1.2633929e-01	 1.3915023e+00	 1.4590891e-01	[ 1.2677906e+00]	 1.8912745e-01


.. parsed-literal::

      72	 1.3477801e+00	 1.2595654e-01	 1.3976824e+00	 1.4538277e-01	[ 1.2687580e+00]	 1.9997716e-01


.. parsed-literal::

      73	 1.3551655e+00	 1.2545950e-01	 1.4051602e+00	 1.4492535e-01	[ 1.2714688e+00]	 2.1685791e-01
      74	 1.3620078e+00	 1.2498045e-01	 1.4121444e+00	 1.4466549e-01	[ 1.2734446e+00]	 1.9675636e-01


.. parsed-literal::

      75	 1.3640185e+00	 1.2454172e-01	 1.4148027e+00	 1.4569506e-01	  1.2632548e+00 	 2.0620680e-01
      76	 1.3730671e+00	 1.2416603e-01	 1.4235267e+00	 1.4504035e-01	[ 1.2765852e+00]	 1.8691015e-01


.. parsed-literal::

      77	 1.3759609e+00	 1.2407701e-01	 1.4263672e+00	 1.4491030e-01	[ 1.2793280e+00]	 2.1359372e-01
      78	 1.3816812e+00	 1.2354744e-01	 1.4322112e+00	 1.4434934e-01	[ 1.2826888e+00]	 1.9517899e-01


.. parsed-literal::

      79	 1.3868166e+00	 1.2318778e-01	 1.4374557e+00	 1.4320787e-01	[ 1.2830829e+00]	 1.8436408e-01
      80	 1.3912288e+00	 1.2288386e-01	 1.4419197e+00	 1.4299884e-01	[ 1.2869017e+00]	 1.8512869e-01


.. parsed-literal::

      81	 1.3946374e+00	 1.2271907e-01	 1.4454018e+00	 1.4282995e-01	[ 1.2897333e+00]	 2.1164799e-01


.. parsed-literal::

      82	 1.3990856e+00	 1.2254679e-01	 1.4501174e+00	 1.4273895e-01	[ 1.2919530e+00]	 2.0121980e-01


.. parsed-literal::

      83	 1.4030227e+00	 1.2223993e-01	 1.4544183e+00	 1.4291135e-01	  1.2913316e+00 	 2.1459937e-01


.. parsed-literal::

      84	 1.4059039e+00	 1.2215035e-01	 1.4573087e+00	 1.4290415e-01	[ 1.2922517e+00]	 2.1401119e-01
      85	 1.4101879e+00	 1.2191840e-01	 1.4617105e+00	 1.4280800e-01	  1.2917608e+00 	 2.0330834e-01


.. parsed-literal::

      86	 1.4136089e+00	 1.2173051e-01	 1.4652069e+00	 1.4282352e-01	  1.2895115e+00 	 1.8911338e-01


.. parsed-literal::

      87	 1.4179699e+00	 1.2144941e-01	 1.4695175e+00	 1.4251529e-01	  1.2896309e+00 	 2.1186900e-01


.. parsed-literal::

      88	 1.4218356e+00	 1.2113935e-01	 1.4734078e+00	 1.4215205e-01	  1.2884167e+00 	 2.0721555e-01


.. parsed-literal::

      89	 1.4248351e+00	 1.2097847e-01	 1.4764473e+00	 1.4224816e-01	  1.2879345e+00 	 2.1957302e-01


.. parsed-literal::

      90	 1.4277817e+00	 1.2066087e-01	 1.4795098e+00	 1.4228651e-01	  1.2875107e+00 	 2.1163154e-01


.. parsed-literal::

      91	 1.4308490e+00	 1.2068043e-01	 1.4825988e+00	 1.4242463e-01	  1.2878197e+00 	 2.1015191e-01
      92	 1.4334142e+00	 1.2061313e-01	 1.4851363e+00	 1.4260026e-01	  1.2893523e+00 	 1.8301797e-01


.. parsed-literal::

      93	 1.4375246e+00	 1.2046560e-01	 1.4892494e+00	 1.4301454e-01	  1.2906983e+00 	 1.8947816e-01


.. parsed-literal::

      94	 1.4403572e+00	 1.2056866e-01	 1.4920891e+00	 1.4403213e-01	[ 1.2945234e+00]	 2.0538044e-01
      95	 1.4438518e+00	 1.2043308e-01	 1.4954713e+00	 1.4402997e-01	[ 1.2979748e+00]	 1.8742347e-01


.. parsed-literal::

      96	 1.4463172e+00	 1.2025242e-01	 1.4978743e+00	 1.4391339e-01	[ 1.3010554e+00]	 2.0625234e-01


.. parsed-literal::

      97	 1.4491128e+00	 1.2004625e-01	 1.5006949e+00	 1.4407365e-01	[ 1.3042515e+00]	 2.0722651e-01


.. parsed-literal::

      98	 1.4524087e+00	 1.1974090e-01	 1.5040579e+00	 1.4387231e-01	[ 1.3087529e+00]	 2.0305634e-01


.. parsed-literal::

      99	 1.4552447e+00	 1.1947599e-01	 1.5069862e+00	 1.4345444e-01	[ 1.3099824e+00]	 2.0804906e-01
     100	 1.4584756e+00	 1.1923301e-01	 1.5103219e+00	 1.4311068e-01	[ 1.3106231e+00]	 1.7659497e-01


.. parsed-literal::

     101	 1.4618589e+00	 1.1905787e-01	 1.5137979e+00	 1.4271140e-01	[ 1.3106606e+00]	 2.1443510e-01


.. parsed-literal::

     102	 1.4655428e+00	 1.1891060e-01	 1.5174899e+00	 1.4221777e-01	[ 1.3118411e+00]	 2.0694876e-01
     103	 1.4682851e+00	 1.1878169e-01	 1.5201997e+00	 1.4235806e-01	[ 1.3133995e+00]	 2.0079589e-01


.. parsed-literal::

     104	 1.4708142e+00	 1.1871291e-01	 1.5226564e+00	 1.4223790e-01	[ 1.3157815e+00]	 2.1458864e-01
     105	 1.4731657e+00	 1.1871115e-01	 1.5249519e+00	 1.4213913e-01	[ 1.3171208e+00]	 1.9778895e-01


.. parsed-literal::

     106	 1.4754233e+00	 1.1864156e-01	 1.5272394e+00	 1.4234515e-01	[ 1.3179423e+00]	 2.1097159e-01


.. parsed-literal::

     107	 1.4775537e+00	 1.1861944e-01	 1.5294820e+00	 1.4216934e-01	[ 1.3180712e+00]	 2.0858002e-01


.. parsed-literal::

     108	 1.4802883e+00	 1.1857400e-01	 1.5324323e+00	 1.4179225e-01	  1.3154539e+00 	 2.0822072e-01


.. parsed-literal::

     109	 1.4825572e+00	 1.1842430e-01	 1.5348771e+00	 1.4158544e-01	  1.3145788e+00 	 2.0503068e-01


.. parsed-literal::

     110	 1.4847378e+00	 1.1838244e-01	 1.5370375e+00	 1.4153386e-01	  1.3146512e+00 	 2.0824671e-01
     111	 1.4865145e+00	 1.1832937e-01	 1.5387581e+00	 1.4144674e-01	  1.3162708e+00 	 1.7319322e-01


.. parsed-literal::

     112	 1.4889060e+00	 1.1824322e-01	 1.5411165e+00	 1.4129278e-01	[ 1.3189895e+00]	 2.0109868e-01


.. parsed-literal::

     113	 1.4903414e+00	 1.1822979e-01	 1.5425539e+00	 1.4112221e-01	[ 1.3206414e+00]	 3.0347466e-01


.. parsed-literal::

     114	 1.4920398e+00	 1.1819684e-01	 1.5442480e+00	 1.4095235e-01	[ 1.3228959e+00]	 2.1905875e-01
     115	 1.4935299e+00	 1.1814167e-01	 1.5457734e+00	 1.4087509e-01	[ 1.3235830e+00]	 2.0493770e-01


.. parsed-literal::

     116	 1.4952228e+00	 1.1807629e-01	 1.5475542e+00	 1.4096272e-01	  1.3233082e+00 	 2.0252919e-01
     117	 1.4970984e+00	 1.1799858e-01	 1.5495884e+00	 1.4108517e-01	  1.3186994e+00 	 1.9376755e-01


.. parsed-literal::

     118	 1.4989578e+00	 1.1795049e-01	 1.5515476e+00	 1.4130228e-01	  1.3159227e+00 	 1.9394159e-01
     119	 1.5012289e+00	 1.1788152e-01	 1.5539574e+00	 1.4164058e-01	  1.3108650e+00 	 1.9584608e-01


.. parsed-literal::

     120	 1.5030469e+00	 1.1783470e-01	 1.5557858e+00	 1.4162868e-01	  1.3089666e+00 	 2.0739436e-01


.. parsed-literal::

     121	 1.5048892e+00	 1.1782203e-01	 1.5575846e+00	 1.4157374e-01	  1.3107864e+00 	 2.0272446e-01


.. parsed-literal::

     122	 1.5061886e+00	 1.1775471e-01	 1.5588998e+00	 1.4150402e-01	  1.3122247e+00 	 2.1876383e-01


.. parsed-literal::

     123	 1.5077305e+00	 1.1779035e-01	 1.5603509e+00	 1.4142449e-01	  1.3169718e+00 	 2.0776987e-01
     124	 1.5087195e+00	 1.1776078e-01	 1.5613133e+00	 1.4148064e-01	  1.3196640e+00 	 2.0046329e-01


.. parsed-literal::

     125	 1.5101270e+00	 1.1767982e-01	 1.5627292e+00	 1.4158830e-01	  1.3210997e+00 	 2.1437335e-01


.. parsed-literal::

     126	 1.5114677e+00	 1.1757009e-01	 1.5641022e+00	 1.4203235e-01	  1.3220268e+00 	 2.0926094e-01


.. parsed-literal::

     127	 1.5133134e+00	 1.1748115e-01	 1.5659657e+00	 1.4210214e-01	  1.3213239e+00 	 2.2749019e-01
     128	 1.5149556e+00	 1.1741426e-01	 1.5676263e+00	 1.4225888e-01	  1.3200822e+00 	 1.7379928e-01


.. parsed-literal::

     129	 1.5163732e+00	 1.1738912e-01	 1.5690630e+00	 1.4253767e-01	  1.3202749e+00 	 2.0553732e-01


.. parsed-literal::

     130	 1.5181785e+00	 1.1735844e-01	 1.5709298e+00	 1.4304282e-01	  1.3193470e+00 	 2.0794725e-01
     131	 1.5199989e+00	 1.1744158e-01	 1.5727636e+00	 1.4336123e-01	  1.3234007e+00 	 1.8946505e-01


.. parsed-literal::

     132	 1.5211926e+00	 1.1740937e-01	 1.5739308e+00	 1.4335730e-01	[ 1.3255461e+00]	 2.0820212e-01


.. parsed-literal::

     133	 1.5228673e+00	 1.1734734e-01	 1.5756019e+00	 1.4320507e-01	[ 1.3261190e+00]	 2.1596622e-01


.. parsed-literal::

     134	 1.5241805e+00	 1.1724263e-01	 1.5770115e+00	 1.4304037e-01	[ 1.3267439e+00]	 2.1251106e-01


.. parsed-literal::

     135	 1.5258208e+00	 1.1720845e-01	 1.5786305e+00	 1.4280707e-01	  1.3260598e+00 	 2.1033859e-01
     136	 1.5274310e+00	 1.1714558e-01	 1.5802875e+00	 1.4245829e-01	  1.3244264e+00 	 2.0389891e-01


.. parsed-literal::

     137	 1.5284754e+00	 1.1708756e-01	 1.5813764e+00	 1.4218631e-01	  1.3244972e+00 	 2.1269989e-01


.. parsed-literal::

     138	 1.5298754e+00	 1.1694663e-01	 1.5828778e+00	 1.4193849e-01	  1.3233413e+00 	 3.1395864e-01


.. parsed-literal::

     139	 1.5316186e+00	 1.1689891e-01	 1.5847160e+00	 1.4151927e-01	  1.3248083e+00 	 2.1303344e-01


.. parsed-literal::

     140	 1.5326493e+00	 1.1686947e-01	 1.5857645e+00	 1.4151138e-01	  1.3248303e+00 	 2.0176578e-01
     141	 1.5347072e+00	 1.1675820e-01	 1.5879234e+00	 1.4156345e-01	  1.3223532e+00 	 1.8495560e-01


.. parsed-literal::

     142	 1.5360164e+00	 1.1665116e-01	 1.5893585e+00	 1.4172045e-01	  1.3172705e+00 	 2.1585584e-01
     143	 1.5375926e+00	 1.1655379e-01	 1.5909038e+00	 1.4176771e-01	  1.3154098e+00 	 1.8860579e-01


.. parsed-literal::

     144	 1.5390292e+00	 1.1646779e-01	 1.5923636e+00	 1.4176373e-01	  1.3125948e+00 	 2.1497726e-01


.. parsed-literal::

     145	 1.5398702e+00	 1.1641900e-01	 1.5932214e+00	 1.4190586e-01	  1.3095771e+00 	 2.0605731e-01


.. parsed-literal::

     146	 1.5413085e+00	 1.1644132e-01	 1.5946688e+00	 1.4216523e-01	  1.3085550e+00 	 2.1064448e-01
     147	 1.5426893e+00	 1.1644843e-01	 1.5960695e+00	 1.4244088e-01	  1.3078533e+00 	 2.0582294e-01


.. parsed-literal::

     148	 1.5443620e+00	 1.1648494e-01	 1.5977978e+00	 1.4270998e-01	  1.3079034e+00 	 2.0632339e-01


.. parsed-literal::

     149	 1.5457383e+00	 1.1648892e-01	 1.5992139e+00	 1.4290954e-01	  1.3083058e+00 	 2.0687318e-01


.. parsed-literal::

     150	 1.5469739e+00	 1.1644489e-01	 1.6004291e+00	 1.4285779e-01	  1.3078165e+00 	 2.1277189e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.02 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f7304b876a0>



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
    CPU times: user 1.75 s, sys: 44 ms, total: 1.79 s
    Wall time: 561 ms


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

