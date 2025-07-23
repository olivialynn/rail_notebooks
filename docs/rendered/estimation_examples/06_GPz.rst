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
       1	-3.5592563e-01	 3.2455522e-01	-3.4626374e-01	 3.0473347e-01	[-3.0939224e-01]	 4.5934844e-01


.. parsed-literal::

       2	-2.8838940e-01	 3.1544274e-01	-2.6597823e-01	 2.9405579e-01	[-2.0324390e-01]	 2.3182297e-01


.. parsed-literal::

       3	-2.4385944e-01	 2.9478657e-01	-2.0103032e-01	 2.7475657e-01	[-1.2449582e-01]	 2.7573037e-01


.. parsed-literal::

       4	-2.1196864e-01	 2.7847664e-01	-1.6418227e-01	 2.6277873e-01	[-8.3054964e-02]	 3.0647469e-01


.. parsed-literal::

       5	-1.4919844e-01	 2.6223904e-01	-1.0971007e-01	 2.4699987e-01	[-2.7835888e-02]	 2.0669842e-01


.. parsed-literal::

       6	-8.7976034e-02	 2.5599687e-01	-5.7393827e-02	 2.4022265e-01	[ 6.5998448e-03]	 2.0571256e-01


.. parsed-literal::

       7	-6.8542287e-02	 2.5296693e-01	-4.4014594e-02	 2.3588374e-01	[ 2.0276636e-02]	 2.1444583e-01
       8	-5.2704075e-02	 2.5013915e-01	-3.3452963e-02	 2.3310113e-01	[ 3.4436633e-02]	 1.9619942e-01


.. parsed-literal::

       9	-4.4331629e-02	 2.4866007e-01	-2.7103510e-02	 2.3147166e-01	[ 3.8143043e-02]	 2.1350861e-01


.. parsed-literal::

      10	-3.3310397e-02	 2.4652230e-01	-1.7925530e-02	 2.2937533e-01	[ 5.0902731e-02]	 2.1824670e-01


.. parsed-literal::

      11	-2.5304106e-02	 2.4521218e-01	-1.1583246e-02	 2.2802239e-01	[ 5.4034558e-02]	 2.0452285e-01


.. parsed-literal::

      12	-2.1128085e-02	 2.4437835e-01	-7.5072842e-03	 2.2721062e-01	[ 5.9328005e-02]	 2.2078156e-01
      13	-1.7477324e-02	 2.4371021e-01	-3.8437864e-03	 2.2619434e-01	[ 6.6316960e-02]	 1.8836021e-01


.. parsed-literal::

      14	 1.3553481e-01	 2.2858687e-01	 1.5799663e-01	 2.0992865e-01	[ 2.0309968e-01]	 4.3369031e-01


.. parsed-literal::

      15	 2.1265150e-01	 2.2235560e-01	 2.3748987e-01	 2.0272167e-01	[ 2.9058414e-01]	 2.1774769e-01
      16	 2.6773428e-01	 2.1583026e-01	 2.9911382e-01	 2.0046692e-01	[ 3.7182782e-01]	 1.8444490e-01


.. parsed-literal::

      17	 3.2187540e-01	 2.1751734e-01	 3.5321889e-01	 2.0638338e-01	[ 3.9896370e-01]	 2.0715523e-01
      18	 3.6368673e-01	 2.1325969e-01	 3.9490409e-01	 2.0359108e-01	[ 4.3486733e-01]	 1.8150163e-01


.. parsed-literal::

      19	 4.3633890e-01	 2.0851530e-01	 4.6785039e-01	 1.9907506e-01	[ 5.0406518e-01]	 2.0931649e-01


.. parsed-literal::

      20	 5.5819780e-01	 2.0371034e-01	 5.9337321e-01	 1.9072144e-01	[ 6.2193864e-01]	 2.0445442e-01


.. parsed-literal::

      21	 6.2363365e-01	 2.0547812e-01	 6.6171861e-01	 1.8737827e-01	[ 6.8212883e-01]	 2.1705055e-01


.. parsed-literal::

      22	 6.6301078e-01	 1.9795639e-01	 7.0195175e-01	 1.8232438e-01	[ 7.0914762e-01]	 2.1220994e-01


.. parsed-literal::

      23	 7.1413516e-01	 1.9508153e-01	 7.5165254e-01	 1.7851888e-01	[ 7.6267992e-01]	 2.0638704e-01
      24	 7.5968834e-01	 1.9782093e-01	 7.9826529e-01	 1.8002250e-01	[ 8.0299500e-01]	 1.9799280e-01


.. parsed-literal::

      25	 7.9673480e-01	 1.9450341e-01	 8.3487269e-01	 1.7478208e-01	[ 8.3747648e-01]	 2.0591664e-01


.. parsed-literal::

      26	 8.1798017e-01	 1.9000918e-01	 8.5675999e-01	 1.7163092e-01	[ 8.5569774e-01]	 2.0824361e-01
      27	 8.5193290e-01	 1.8591597e-01	 8.9262884e-01	 1.6807339e-01	[ 8.8627094e-01]	 1.8055129e-01


.. parsed-literal::

      28	 8.7501534e-01	 1.8368318e-01	 9.1594074e-01	 1.6496876e-01	[ 9.1220028e-01]	 2.0863104e-01
      29	 8.9876883e-01	 1.8094818e-01	 9.3957948e-01	 1.6241366e-01	[ 9.3695150e-01]	 1.9834638e-01


.. parsed-literal::

      30	 9.1383263e-01	 1.7982324e-01	 9.5540999e-01	 1.6104450e-01	[ 9.5038688e-01]	 2.1034145e-01


.. parsed-literal::

      31	 9.3651032e-01	 1.7774722e-01	 9.7920199e-01	 1.5844388e-01	[ 9.6891824e-01]	 2.0984340e-01


.. parsed-literal::

      32	 9.5084957e-01	 1.7685835e-01	 9.9471285e-01	 1.5661973e-01	[ 9.9221415e-01]	 2.1236181e-01
      33	 9.6838796e-01	 1.7488720e-01	 1.0119743e+00	 1.5572941e-01	[ 1.0061404e+00]	 2.0054722e-01


.. parsed-literal::

      34	 9.8143158e-01	 1.7369352e-01	 1.0250427e+00	 1.5480631e-01	[ 1.0184085e+00]	 2.0454097e-01
      35	 9.9712167e-01	 1.7251236e-01	 1.0411150e+00	 1.5322458e-01	[ 1.0349797e+00]	 1.9996476e-01


.. parsed-literal::

      36	 1.0187636e+00	 1.7160862e-01	 1.0643900e+00	 1.5093955e-01	[ 1.0577134e+00]	 1.9951153e-01


.. parsed-literal::

      37	 1.0350686e+00	 1.6935269e-01	 1.0815005e+00	 1.4939573e-01	[ 1.0679619e+00]	 2.0568967e-01


.. parsed-literal::

      38	 1.0464875e+00	 1.6836510e-01	 1.0927646e+00	 1.4872611e-01	[ 1.0854334e+00]	 2.1098685e-01


.. parsed-literal::

      39	 1.0574974e+00	 1.6715832e-01	 1.1043736e+00	 1.4864466e-01	[ 1.0929824e+00]	 2.1785426e-01
      40	 1.0673244e+00	 1.6666961e-01	 1.1145692e+00	 1.4907850e-01	[ 1.0969196e+00]	 1.8534589e-01


.. parsed-literal::

      41	 1.0790559e+00	 1.6542798e-01	 1.1263894e+00	 1.4864267e-01	[ 1.1012441e+00]	 1.8244839e-01


.. parsed-literal::

      42	 1.0938244e+00	 1.6346054e-01	 1.1414225e+00	 1.4744922e-01	[ 1.1091425e+00]	 2.1593690e-01
      43	 1.0967384e+00	 1.6452867e-01	 1.1445819e+00	 1.4588461e-01	[ 1.1273205e+00]	 1.9819832e-01


.. parsed-literal::

      44	 1.1125491e+00	 1.6099855e-01	 1.1601423e+00	 1.4409507e-01	[ 1.1376116e+00]	 2.1229792e-01
      45	 1.1198942e+00	 1.5972732e-01	 1.1675481e+00	 1.4373002e-01	[ 1.1410312e+00]	 1.9955373e-01


.. parsed-literal::

      46	 1.1272316e+00	 1.5870726e-01	 1.1751093e+00	 1.4270299e-01	[ 1.1483386e+00]	 2.1660113e-01
      47	 1.1445345e+00	 1.5600074e-01	 1.1927392e+00	 1.3939235e-01	[ 1.1620423e+00]	 1.7121768e-01


.. parsed-literal::

      48	 1.1525788e+00	 1.5520831e-01	 1.2012263e+00	 1.3716524e-01	[ 1.1724439e+00]	 3.1909013e-01


.. parsed-literal::

      49	 1.1635842e+00	 1.5368376e-01	 1.2123255e+00	 1.3588405e-01	[ 1.1795665e+00]	 2.1059370e-01
      50	 1.1772006e+00	 1.5200742e-01	 1.2259084e+00	 1.3502901e-01	[ 1.1887319e+00]	 2.0155525e-01


.. parsed-literal::

      51	 1.1897982e+00	 1.5050982e-01	 1.2386818e+00	 1.3439206e-01	[ 1.1982081e+00]	 1.9968104e-01
      52	 1.1932043e+00	 1.4882975e-01	 1.2426808e+00	 1.3467750e-01	  1.1911290e+00 	 1.9951129e-01


.. parsed-literal::

      53	 1.2060808e+00	 1.4841173e-01	 1.2551127e+00	 1.3393873e-01	[ 1.2074881e+00]	 2.0463371e-01


.. parsed-literal::

      54	 1.2106963e+00	 1.4818164e-01	 1.2598146e+00	 1.3367966e-01	[ 1.2109612e+00]	 2.1727514e-01
      55	 1.2209286e+00	 1.4748968e-01	 1.2703001e+00	 1.3329241e-01	[ 1.2162282e+00]	 1.7833114e-01


.. parsed-literal::

      56	 1.2343953e+00	 1.4684206e-01	 1.2839899e+00	 1.3307171e-01	[ 1.2219263e+00]	 2.0577407e-01


.. parsed-literal::

      57	 1.2426641e+00	 1.4555078e-01	 1.2926814e+00	 1.3264565e-01	[ 1.2253077e+00]	 3.2760882e-01
      58	 1.2513288e+00	 1.4500968e-01	 1.3015015e+00	 1.3236348e-01	[ 1.2310706e+00]	 1.9092965e-01


.. parsed-literal::

      59	 1.2592047e+00	 1.4450307e-01	 1.3094644e+00	 1.3145094e-01	[ 1.2402901e+00]	 2.1089530e-01


.. parsed-literal::

      60	 1.2693533e+00	 1.4237110e-01	 1.3203754e+00	 1.3013722e-01	[ 1.2574560e+00]	 2.0698237e-01


.. parsed-literal::

      61	 1.2751980e+00	 1.4260398e-01	 1.3264232e+00	 1.2922235e-01	[ 1.2740346e+00]	 2.0269465e-01


.. parsed-literal::

      62	 1.2800152e+00	 1.4213722e-01	 1.3309717e+00	 1.2910016e-01	[ 1.2761626e+00]	 2.1505857e-01
      63	 1.2860676e+00	 1.4114543e-01	 1.3371221e+00	 1.2907135e-01	[ 1.2779342e+00]	 1.9918704e-01


.. parsed-literal::

      64	 1.2937893e+00	 1.4021309e-01	 1.3451586e+00	 1.2908570e-01	[ 1.2837437e+00]	 2.0281196e-01


.. parsed-literal::

      65	 1.3031776e+00	 1.3927243e-01	 1.3545509e+00	 1.2919332e-01	[ 1.2964364e+00]	 2.1100831e-01


.. parsed-literal::

      66	 1.3080669e+00	 1.3837668e-01	 1.3597681e+00	 1.2933605e-01	[ 1.3111471e+00]	 2.1286273e-01


.. parsed-literal::

      67	 1.3171844e+00	 1.3799272e-01	 1.3685294e+00	 1.2849691e-01	[ 1.3180100e+00]	 2.0868444e-01


.. parsed-literal::

      68	 1.3210991e+00	 1.3739702e-01	 1.3724896e+00	 1.2813764e-01	[ 1.3206085e+00]	 2.1227431e-01
      69	 1.3275261e+00	 1.3625555e-01	 1.3791939e+00	 1.2772824e-01	[ 1.3218114e+00]	 1.8133998e-01


.. parsed-literal::

      70	 1.3342675e+00	 1.3495755e-01	 1.3857696e+00	 1.2754574e-01	  1.3206868e+00 	 2.1515036e-01


.. parsed-literal::

      71	 1.3418200e+00	 1.3432901e-01	 1.3934155e+00	 1.2778385e-01	[ 1.3252692e+00]	 2.1501851e-01


.. parsed-literal::

      72	 1.3481827e+00	 1.3400699e-01	 1.3998045e+00	 1.2785390e-01	[ 1.3263762e+00]	 2.1315145e-01


.. parsed-literal::

      73	 1.3533161e+00	 1.3362384e-01	 1.4050179e+00	 1.2828101e-01	[ 1.3265393e+00]	 2.1600127e-01


.. parsed-literal::

      74	 1.3577919e+00	 1.3332131e-01	 1.4093860e+00	 1.2847093e-01	  1.3258514e+00 	 2.1017671e-01


.. parsed-literal::

      75	 1.3634918e+00	 1.3292128e-01	 1.4150185e+00	 1.2837239e-01	[ 1.3298218e+00]	 2.0858550e-01


.. parsed-literal::

      76	 1.3681055e+00	 1.3258943e-01	 1.4196733e+00	 1.2858137e-01	[ 1.3310841e+00]	 2.2589421e-01


.. parsed-literal::

      77	 1.3719704e+00	 1.3218758e-01	 1.4236555e+00	 1.2850441e-01	[ 1.3323349e+00]	 2.1165061e-01


.. parsed-literal::

      78	 1.3752882e+00	 1.3183668e-01	 1.4272957e+00	 1.2919364e-01	  1.3276005e+00 	 2.2325039e-01


.. parsed-literal::

      79	 1.3801913e+00	 1.3149949e-01	 1.4322088e+00	 1.2896951e-01	[ 1.3327441e+00]	 2.1020865e-01


.. parsed-literal::

      80	 1.3825775e+00	 1.3132982e-01	 1.4345766e+00	 1.2871249e-01	[ 1.3384955e+00]	 2.0950127e-01


.. parsed-literal::

      81	 1.3869034e+00	 1.3103199e-01	 1.4390722e+00	 1.2884545e-01	[ 1.3427861e+00]	 2.2633910e-01
      82	 1.3900064e+00	 1.3072519e-01	 1.4423415e+00	 1.2891934e-01	[ 1.3579108e+00]	 1.8349814e-01


.. parsed-literal::

      83	 1.3942509e+00	 1.3057256e-01	 1.4464795e+00	 1.2898253e-01	[ 1.3582450e+00]	 1.9830871e-01
      84	 1.3976224e+00	 1.3039863e-01	 1.4498476e+00	 1.2905685e-01	[ 1.3584655e+00]	 1.8966746e-01


.. parsed-literal::

      85	 1.4008043e+00	 1.3016615e-01	 1.4531381e+00	 1.2902687e-01	[ 1.3591791e+00]	 2.0445275e-01
      86	 1.4073091e+00	 1.2956928e-01	 1.4598331e+00	 1.2877978e-01	[ 1.3594721e+00]	 1.9961834e-01


.. parsed-literal::

      87	 1.4113553e+00	 1.2900779e-01	 1.4639695e+00	 1.2850178e-01	[ 1.3631081e+00]	 3.2381320e-01
      88	 1.4174842e+00	 1.2836490e-01	 1.4701269e+00	 1.2794797e-01	[ 1.3655369e+00]	 1.8881321e-01


.. parsed-literal::

      89	 1.4216488e+00	 1.2791388e-01	 1.4743440e+00	 1.2742407e-01	[ 1.3695631e+00]	 2.0146132e-01


.. parsed-literal::

      90	 1.4246158e+00	 1.2766126e-01	 1.4773298e+00	 1.2712360e-01	[ 1.3704764e+00]	 2.1022224e-01


.. parsed-literal::

      91	 1.4281294e+00	 1.2740872e-01	 1.4807958e+00	 1.2660104e-01	[ 1.3750691e+00]	 2.1911907e-01


.. parsed-literal::

      92	 1.4315846e+00	 1.2724346e-01	 1.4843588e+00	 1.2620586e-01	[ 1.3803731e+00]	 2.1674609e-01
      93	 1.4346423e+00	 1.2705077e-01	 1.4874222e+00	 1.2603843e-01	[ 1.3836637e+00]	 1.8981791e-01


.. parsed-literal::

      94	 1.4367735e+00	 1.2689414e-01	 1.4895780e+00	 1.2606652e-01	[ 1.3850702e+00]	 2.0978022e-01


.. parsed-literal::

      95	 1.4408802e+00	 1.2662478e-01	 1.4938565e+00	 1.2630390e-01	  1.3849160e+00 	 2.0602417e-01


.. parsed-literal::

      96	 1.4432545e+00	 1.2635007e-01	 1.4963241e+00	 1.2615800e-01	[ 1.3865023e+00]	 3.2533765e-01
      97	 1.4460118e+00	 1.2628360e-01	 1.4991592e+00	 1.2636587e-01	[ 1.3868676e+00]	 1.8263054e-01


.. parsed-literal::

      98	 1.4487719e+00	 1.2623788e-01	 1.5020257e+00	 1.2640650e-01	[ 1.3878434e+00]	 2.1774006e-01


.. parsed-literal::

      99	 1.4511285e+00	 1.2625477e-01	 1.5045566e+00	 1.2647398e-01	[ 1.3910956e+00]	 2.1697640e-01
     100	 1.4537623e+00	 1.2595961e-01	 1.5073465e+00	 1.2616186e-01	[ 1.3932603e+00]	 1.8487716e-01


.. parsed-literal::

     101	 1.4562878e+00	 1.2573687e-01	 1.5099601e+00	 1.2590865e-01	[ 1.3981286e+00]	 2.1929789e-01


.. parsed-literal::

     102	 1.4588289e+00	 1.2539630e-01	 1.5126370e+00	 1.2573087e-01	[ 1.4042484e+00]	 2.1040201e-01


.. parsed-literal::

     103	 1.4610212e+00	 1.2517430e-01	 1.5148711e+00	 1.2564844e-01	  1.4038239e+00 	 2.1752882e-01


.. parsed-literal::

     104	 1.4630024e+00	 1.2502569e-01	 1.5167732e+00	 1.2555899e-01	[ 1.4054685e+00]	 2.1124005e-01


.. parsed-literal::

     105	 1.4659509e+00	 1.2475821e-01	 1.5196679e+00	 1.2558724e-01	  1.4049206e+00 	 2.1767545e-01


.. parsed-literal::

     106	 1.4677362e+00	 1.2457900e-01	 1.5214404e+00	 1.2535312e-01	[ 1.4089304e+00]	 2.1574569e-01


.. parsed-literal::

     107	 1.4694196e+00	 1.2451802e-01	 1.5230751e+00	 1.2523959e-01	[ 1.4113584e+00]	 2.0787263e-01
     108	 1.4723362e+00	 1.2436839e-01	 1.5260216e+00	 1.2503649e-01	[ 1.4165243e+00]	 1.8751216e-01


.. parsed-literal::

     109	 1.4738833e+00	 1.2408763e-01	 1.5275854e+00	 1.2459776e-01	[ 1.4201257e+00]	 2.2474194e-01


.. parsed-literal::

     110	 1.4753351e+00	 1.2403962e-01	 1.5290278e+00	 1.2457011e-01	[ 1.4215231e+00]	 2.1463799e-01


.. parsed-literal::

     111	 1.4780281e+00	 1.2381679e-01	 1.5317463e+00	 1.2434925e-01	[ 1.4238807e+00]	 2.0480514e-01


.. parsed-literal::

     112	 1.4797397e+00	 1.2366915e-01	 1.5334562e+00	 1.2421401e-01	[ 1.4245033e+00]	 2.0102572e-01
     113	 1.4826846e+00	 1.2279393e-01	 1.5364738e+00	 1.2346576e-01	  1.4196931e+00 	 1.7399168e-01


.. parsed-literal::

     114	 1.4852844e+00	 1.2272311e-01	 1.5390201e+00	 1.2349090e-01	  1.4198676e+00 	 2.0173883e-01


.. parsed-literal::

     115	 1.4864820e+00	 1.2276317e-01	 1.5401901e+00	 1.2347800e-01	  1.4219433e+00 	 2.0912790e-01


.. parsed-literal::

     116	 1.4884355e+00	 1.2266262e-01	 1.5422028e+00	 1.2329677e-01	  1.4234021e+00 	 2.0829105e-01
     117	 1.4903693e+00	 1.2237014e-01	 1.5442984e+00	 1.2299046e-01	  1.4197075e+00 	 2.0477724e-01


.. parsed-literal::

     118	 1.4918638e+00	 1.2210520e-01	 1.5459586e+00	 1.2244684e-01	  1.4215191e+00 	 2.0686603e-01


.. parsed-literal::

     119	 1.4934090e+00	 1.2198751e-01	 1.5474840e+00	 1.2237966e-01	  1.4217530e+00 	 2.1945190e-01
     120	 1.4947628e+00	 1.2186069e-01	 1.5488657e+00	 1.2219154e-01	  1.4226190e+00 	 1.9036508e-01


.. parsed-literal::

     121	 1.4961484e+00	 1.2176849e-01	 1.5502909e+00	 1.2196991e-01	  1.4232905e+00 	 2.0082688e-01


.. parsed-literal::

     122	 1.4975484e+00	 1.2163145e-01	 1.5518420e+00	 1.2167699e-01	  1.4217084e+00 	 2.2007680e-01
     123	 1.4997171e+00	 1.2152327e-01	 1.5540064e+00	 1.2140549e-01	  1.4240939e+00 	 1.9671464e-01


.. parsed-literal::

     124	 1.5006803e+00	 1.2144477e-01	 1.5549577e+00	 1.2136836e-01	[ 1.4247273e+00]	 2.1456766e-01


.. parsed-literal::

     125	 1.5021914e+00	 1.2126357e-01	 1.5565245e+00	 1.2118851e-01	[ 1.4250014e+00]	 2.1110177e-01


.. parsed-literal::

     126	 1.5035227e+00	 1.2106077e-01	 1.5579439e+00	 1.2104851e-01	  1.4246837e+00 	 2.0733166e-01


.. parsed-literal::

     127	 1.5052403e+00	 1.2085826e-01	 1.5597114e+00	 1.2080876e-01	[ 1.4254593e+00]	 2.0614338e-01


.. parsed-literal::

     128	 1.5065109e+00	 1.2077173e-01	 1.5610042e+00	 1.2074703e-01	[ 1.4257585e+00]	 2.0685434e-01
     129	 1.5076754e+00	 1.2070345e-01	 1.5621695e+00	 1.2074741e-01	[ 1.4261424e+00]	 1.8912673e-01


.. parsed-literal::

     130	 1.5091622e+00	 1.2061974e-01	 1.5636660e+00	 1.2075038e-01	[ 1.4277018e+00]	 1.9634032e-01


.. parsed-literal::

     131	 1.5109635e+00	 1.2047696e-01	 1.5654510e+00	 1.2076039e-01	[ 1.4281492e+00]	 2.0932221e-01
     132	 1.5127009e+00	 1.2030871e-01	 1.5671360e+00	 1.2067198e-01	[ 1.4310329e+00]	 2.0304513e-01


.. parsed-literal::

     133	 1.5142648e+00	 1.2013203e-01	 1.5686543e+00	 1.2059049e-01	[ 1.4332230e+00]	 2.0494533e-01


.. parsed-literal::

     134	 1.5157857e+00	 1.2001483e-01	 1.5701741e+00	 1.2058584e-01	[ 1.4340901e+00]	 2.0940351e-01


.. parsed-literal::

     135	 1.5165897e+00	 1.1957098e-01	 1.5711096e+00	 1.2027024e-01	  1.4340547e+00 	 2.0999932e-01
     136	 1.5186731e+00	 1.1969423e-01	 1.5731638e+00	 1.2055827e-01	  1.4326518e+00 	 1.9451094e-01


.. parsed-literal::

     137	 1.5194103e+00	 1.1971286e-01	 1.5739112e+00	 1.2060926e-01	  1.4319758e+00 	 2.1336436e-01
     138	 1.5205678e+00	 1.1962544e-01	 1.5751158e+00	 1.2066339e-01	  1.4293432e+00 	 1.9983697e-01


.. parsed-literal::

     139	 1.5217776e+00	 1.1948750e-01	 1.5764123e+00	 1.2070125e-01	  1.4275014e+00 	 1.9592667e-01
     140	 1.5233245e+00	 1.1930145e-01	 1.5779554e+00	 1.2075405e-01	  1.4253958e+00 	 1.9099092e-01


.. parsed-literal::

     141	 1.5245587e+00	 1.1914589e-01	 1.5791732e+00	 1.2080170e-01	  1.4254993e+00 	 2.0027614e-01


.. parsed-literal::

     142	 1.5258835e+00	 1.1900485e-01	 1.5805143e+00	 1.2092920e-01	  1.4246633e+00 	 2.0316458e-01


.. parsed-literal::

     143	 1.5268669e+00	 1.1887276e-01	 1.5815951e+00	 1.2111984e-01	  1.4233882e+00 	 2.0335293e-01


.. parsed-literal::

     144	 1.5286376e+00	 1.1883703e-01	 1.5833625e+00	 1.2121429e-01	  1.4229022e+00 	 2.0948601e-01


.. parsed-literal::

     145	 1.5295850e+00	 1.1882889e-01	 1.5843368e+00	 1.2125452e-01	  1.4217465e+00 	 2.0974779e-01
     146	 1.5305342e+00	 1.1881499e-01	 1.5853433e+00	 1.2129949e-01	  1.4200968e+00 	 1.8903136e-01


.. parsed-literal::

     147	 1.5312521e+00	 1.1872495e-01	 1.5861064e+00	 1.2124286e-01	  1.4186085e+00 	 3.1157327e-01
     148	 1.5321655e+00	 1.1867523e-01	 1.5870469e+00	 1.2120196e-01	  1.4174348e+00 	 1.9202781e-01


.. parsed-literal::

     149	 1.5331914e+00	 1.1858316e-01	 1.5880943e+00	 1.2108397e-01	  1.4169831e+00 	 2.0661831e-01


.. parsed-literal::

     150	 1.5342911e+00	 1.1846906e-01	 1.5892177e+00	 1.2089043e-01	  1.4158833e+00 	 2.1001697e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.11 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f20ecc42260>



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
    CPU times: user 1.67 s, sys: 54.9 ms, total: 1.72 s
    Wall time: 532 ms


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

