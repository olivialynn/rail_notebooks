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
       1	-3.4177579e-01	 3.2020688e-01	-3.3212373e-01	 3.2280864e-01	[-3.3604773e-01]	 4.6620679e-01


.. parsed-literal::

       2	-2.7233886e-01	 3.0987814e-01	-2.4847514e-01	 3.1249200e-01	[-2.5345229e-01]	 2.3120356e-01


.. parsed-literal::

       3	-2.2784943e-01	 2.8863038e-01	-1.8471958e-01	 2.8815204e-01	[-1.8171612e-01]	 2.9356384e-01
       4	-1.9059358e-01	 2.6541371e-01	-1.4909192e-01	 2.6658510e-01	[-1.5614471e-01]	 1.8826985e-01


.. parsed-literal::

       5	-9.9058770e-02	 2.5585235e-01	-6.5826120e-02	 2.6218865e-01	[-8.4890235e-02]	 2.0688629e-01


.. parsed-literal::

       6	-6.6760650e-02	 2.5098550e-01	-3.7388110e-02	 2.5394210e-01	[-4.7959549e-02]	 2.0747733e-01


.. parsed-literal::

       7	-4.9593139e-02	 2.4823532e-01	-2.6022496e-02	 2.5165414e-01	[-3.6950284e-02]	 2.0984483e-01


.. parsed-literal::

       8	-3.7597512e-02	 2.4626594e-01	-1.7735881e-02	 2.4974669e-01	[-2.8133134e-02]	 2.0874476e-01


.. parsed-literal::

       9	-2.5692333e-02	 2.4409453e-01	-8.3582725e-03	 2.4693694e-01	[-1.6674882e-02]	 2.0129180e-01
      10	-1.5703274e-02	 2.4215638e-01	-3.5551747e-04	 2.4449615e-01	[-8.1407604e-03]	 2.0531440e-01


.. parsed-literal::

      11	-1.0518242e-02	 2.4132265e-01	 3.8220433e-03	 2.4279042e-01	[-1.0562902e-03]	 1.9699049e-01
      12	-7.4624354e-03	 2.4081350e-01	 6.7086244e-03	 2.4237049e-01	[ 1.6406602e-03]	 1.7605495e-01


.. parsed-literal::

      13	-3.8112946e-03	 2.4004619e-01	 1.0419088e-02	 2.4217053e-01	[ 4.2744317e-03]	 2.1773839e-01


.. parsed-literal::

      14	 1.1464056e-01	 2.2525512e-01	 1.3654194e-01	 2.3485941e-01	[ 1.2339928e-01]	 4.3704486e-01


.. parsed-literal::

      15	 1.9472005e-01	 2.2494522e-01	 2.2142131e-01	 2.2915727e-01	[ 2.0563555e-01]	 3.1555891e-01
      16	 2.4228227e-01	 2.1930419e-01	 2.6935114e-01	 2.2076915e-01	[ 2.5415788e-01]	 1.7191148e-01


.. parsed-literal::

      17	 3.1116450e-01	 2.1331997e-01	 3.4184976e-01	 2.1613300e-01	[ 3.1556670e-01]	 1.7410135e-01
      18	 3.4414003e-01	 2.1159309e-01	 3.7705776e-01	 2.1375148e-01	[ 3.4681548e-01]	 1.8380761e-01


.. parsed-literal::

      19	 4.0672329e-01	 2.0967457e-01	 4.4111614e-01	 2.1353164e-01	[ 3.9673991e-01]	 2.1139169e-01
      20	 4.4045044e-01	 2.0980200e-01	 4.7454182e-01	 2.1493634e-01	[ 4.3280200e-01]	 1.9669652e-01


.. parsed-literal::

      21	 4.8195692e-01	 2.0884833e-01	 5.1668265e-01	 2.1353485e-01	[ 4.7402402e-01]	 2.0732594e-01
      22	 5.4861728e-01	 2.0739311e-01	 5.8530559e-01	 2.1309076e-01	[ 5.2888271e-01]	 2.0175004e-01


.. parsed-literal::

      23	 5.8499890e-01	 2.0802122e-01	 6.2396518e-01	 2.0923923e-01	[ 5.7415176e-01]	 3.1212997e-01


.. parsed-literal::

      24	 6.1935797e-01	 2.0265678e-01	 6.5892649e-01	 2.0516041e-01	[ 5.9979712e-01]	 2.1139693e-01


.. parsed-literal::

      25	 6.4841892e-01	 2.0118384e-01	 6.8702486e-01	 2.0385172e-01	[ 6.3472558e-01]	 2.0969582e-01
      26	 6.8000149e-01	 2.0077823e-01	 7.1656355e-01	 2.0310749e-01	[ 6.7734232e-01]	 1.9698453e-01


.. parsed-literal::

      27	 6.9952485e-01	 2.0150508e-01	 7.3557406e-01	 2.0354992e-01	[ 6.9806498e-01]	 2.1879530e-01
      28	 7.2471121e-01	 1.9893365e-01	 7.6357279e-01	 2.0131287e-01	[ 7.2372736e-01]	 1.7789006e-01


.. parsed-literal::

      29	 7.4531861e-01	 1.9833827e-01	 7.8389367e-01	 2.0211113e-01	[ 7.4014463e-01]	 2.1329379e-01


.. parsed-literal::

      30	 7.6670059e-01	 1.9986160e-01	 8.0509700e-01	 2.0226213e-01	[ 7.5473708e-01]	 2.1093369e-01


.. parsed-literal::

      31	 7.9108317e-01	 2.0041569e-01	 8.2984071e-01	 2.0146011e-01	[ 7.7181913e-01]	 2.1059823e-01


.. parsed-literal::

      32	 8.3534417e-01	 2.0091493e-01	 8.7640409e-01	 1.9908543e-01	[ 7.9874986e-01]	 2.0699501e-01
      33	 8.3543822e-01	 2.0818973e-01	 8.7700881e-01	 2.1046082e-01	[ 8.0152835e-01]	 1.8458629e-01


.. parsed-literal::

      34	 8.6461020e-01	 2.0084006e-01	 9.0589173e-01	 2.0323758e-01	[ 8.2500377e-01]	 2.0444703e-01


.. parsed-literal::

      35	 8.8253290e-01	 1.9703311e-01	 9.2360327e-01	 1.9774015e-01	[ 8.3733777e-01]	 2.1011615e-01
      36	 8.9688005e-01	 1.9617124e-01	 9.3834336e-01	 1.9665665e-01	[ 8.4804716e-01]	 2.0236516e-01


.. parsed-literal::

      37	 9.2080191e-01	 1.9337903e-01	 9.6341651e-01	 1.9413833e-01	[ 8.6767976e-01]	 2.1021938e-01


.. parsed-literal::

      38	 9.4098253e-01	 1.9203445e-01	 9.8429378e-01	 1.9268439e-01	[ 8.8778571e-01]	 2.1229386e-01


.. parsed-literal::

      39	 9.6449689e-01	 1.8887800e-01	 1.0083199e+00	 1.9070572e-01	[ 9.1167867e-01]	 2.1356797e-01
      40	 9.8632576e-01	 1.8641206e-01	 1.0311102e+00	 1.8624901e-01	[ 9.3436487e-01]	 2.0024014e-01


.. parsed-literal::

      41	 1.0033519e+00	 1.8304683e-01	 1.0489147e+00	 1.8302197e-01	[ 9.4823237e-01]	 2.1203899e-01
      42	 1.0149410e+00	 1.8089477e-01	 1.0607283e+00	 1.8256526e-01	[ 9.5956744e-01]	 1.9822836e-01


.. parsed-literal::

      43	 1.0266697e+00	 1.7909426e-01	 1.0725426e+00	 1.8153929e-01	[ 9.6887760e-01]	 1.9805741e-01


.. parsed-literal::

      44	 1.0437066e+00	 1.7612357e-01	 1.0902638e+00	 1.8040783e-01	[ 9.8696577e-01]	 2.1092486e-01


.. parsed-literal::

      45	 1.0600750e+00	 1.7360968e-01	 1.1065234e+00	 1.7705233e-01	[ 1.0044659e+00]	 2.1535897e-01
      46	 1.0741729e+00	 1.7207074e-01	 1.1205290e+00	 1.7470806e-01	[ 1.0167015e+00]	 1.8933105e-01


.. parsed-literal::

      47	 1.0906069e+00	 1.6925873e-01	 1.1371016e+00	 1.7022508e-01	[ 1.0310776e+00]	 2.1114445e-01


.. parsed-literal::

      48	 1.1072326e+00	 1.6440424e-01	 1.1539109e+00	 1.6477989e-01	[ 1.0440458e+00]	 2.1857524e-01


.. parsed-literal::

      49	 1.1196255e+00	 1.6104759e-01	 1.1660071e+00	 1.6131762e-01	[ 1.0682225e+00]	 2.0797777e-01


.. parsed-literal::

      50	 1.1315847e+00	 1.5964832e-01	 1.1782892e+00	 1.6075556e-01	[ 1.0772490e+00]	 2.0408273e-01


.. parsed-literal::

      51	 1.1498690e+00	 1.5697877e-01	 1.1980297e+00	 1.6059908e-01	[ 1.0860741e+00]	 2.2118545e-01


.. parsed-literal::

      52	 1.1633337e+00	 1.5574465e-01	 1.2114449e+00	 1.5965697e-01	[ 1.1014108e+00]	 2.1182632e-01
      53	 1.1758572e+00	 1.5462948e-01	 1.2242740e+00	 1.5902315e-01	[ 1.1035277e+00]	 1.8896556e-01


.. parsed-literal::

      54	 1.1877256e+00	 1.5342302e-01	 1.2367278e+00	 1.5795678e-01	  1.1024025e+00 	 1.7961955e-01


.. parsed-literal::

      55	 1.2011345e+00	 1.5140943e-01	 1.2509377e+00	 1.5541892e-01	  1.0994533e+00 	 2.1856666e-01


.. parsed-literal::

      56	 1.2125235e+00	 1.4910576e-01	 1.2624577e+00	 1.5279849e-01	[ 1.1110842e+00]	 2.0825624e-01


.. parsed-literal::

      57	 1.2223514e+00	 1.4779559e-01	 1.2721344e+00	 1.5175882e-01	[ 1.1223910e+00]	 2.0786691e-01


.. parsed-literal::

      58	 1.2399327e+00	 1.4440496e-01	 1.2901745e+00	 1.4878126e-01	[ 1.1407740e+00]	 2.2089434e-01


.. parsed-literal::

      59	 1.2515729e+00	 1.4286578e-01	 1.3015695e+00	 1.4786505e-01	[ 1.1631112e+00]	 2.1313953e-01


.. parsed-literal::

      60	 1.2636916e+00	 1.4125834e-01	 1.3138524e+00	 1.4637206e-01	[ 1.1835338e+00]	 2.1152043e-01


.. parsed-literal::

      61	 1.2760446e+00	 1.4014410e-01	 1.3265129e+00	 1.4533781e-01	[ 1.1954848e+00]	 2.0536232e-01


.. parsed-literal::

      62	 1.2885605e+00	 1.3902815e-01	 1.3395145e+00	 1.4469318e-01	[ 1.2098668e+00]	 2.1870136e-01


.. parsed-literal::

      63	 1.3000139e+00	 1.3817894e-01	 1.3508782e+00	 1.4459483e-01	[ 1.2210684e+00]	 2.1974921e-01


.. parsed-literal::

      64	 1.3080275e+00	 1.3743842e-01	 1.3586886e+00	 1.4438261e-01	[ 1.2288704e+00]	 2.1919417e-01
      65	 1.3181499e+00	 1.3628194e-01	 1.3691737e+00	 1.4378011e-01	[ 1.2333477e+00]	 1.7838693e-01


.. parsed-literal::

      66	 1.3238227e+00	 1.3640747e-01	 1.3748180e+00	 1.4389827e-01	[ 1.2393625e+00]	 2.0830917e-01


.. parsed-literal::

      67	 1.3309166e+00	 1.3549413e-01	 1.3818210e+00	 1.4258734e-01	[ 1.2453203e+00]	 2.0547676e-01


.. parsed-literal::

      68	 1.3369113e+00	 1.3495325e-01	 1.3879901e+00	 1.4181307e-01	[ 1.2492810e+00]	 2.0121455e-01


.. parsed-literal::

      69	 1.3437302e+00	 1.3452508e-01	 1.3949992e+00	 1.4126826e-01	[ 1.2540297e+00]	 2.1281004e-01


.. parsed-literal::

      70	 1.3537875e+00	 1.3360050e-01	 1.4053704e+00	 1.4052647e-01	[ 1.2649280e+00]	 2.0294666e-01


.. parsed-literal::

      71	 1.3622403e+00	 1.3320764e-01	 1.4137725e+00	 1.4082275e-01	[ 1.2714038e+00]	 2.1496272e-01
      72	 1.3699247e+00	 1.3242846e-01	 1.4214394e+00	 1.4055044e-01	[ 1.2763163e+00]	 1.7643714e-01


.. parsed-literal::

      73	 1.3784408e+00	 1.3150467e-01	 1.4301277e+00	 1.4048663e-01	[ 1.2783653e+00]	 2.0412207e-01


.. parsed-literal::

      74	 1.3829686e+00	 1.3142008e-01	 1.4351626e+00	 1.4060384e-01	  1.2775852e+00 	 2.1529722e-01


.. parsed-literal::

      75	 1.3912873e+00	 1.3078367e-01	 1.4432106e+00	 1.3971962e-01	[ 1.2839820e+00]	 2.1352530e-01


.. parsed-literal::

      76	 1.3954681e+00	 1.3051593e-01	 1.4474587e+00	 1.3933991e-01	[ 1.2860223e+00]	 2.0557976e-01


.. parsed-literal::

      77	 1.4007991e+00	 1.3025566e-01	 1.4529595e+00	 1.3899666e-01	[ 1.2874584e+00]	 2.1768522e-01


.. parsed-literal::

      78	 1.4083969e+00	 1.3013696e-01	 1.4608140e+00	 1.3831475e-01	[ 1.2883532e+00]	 2.1250463e-01


.. parsed-literal::

      79	 1.4137850e+00	 1.2996476e-01	 1.4663919e+00	 1.3859467e-01	  1.2866188e+00 	 2.1245718e-01


.. parsed-literal::

      80	 1.4190323e+00	 1.2980237e-01	 1.4713945e+00	 1.3834640e-01	[ 1.2922802e+00]	 2.0472026e-01


.. parsed-literal::

      81	 1.4221726e+00	 1.2958981e-01	 1.4745129e+00	 1.3818785e-01	[ 1.2947307e+00]	 2.0960331e-01
      82	 1.4271084e+00	 1.2934710e-01	 1.4795891e+00	 1.3801452e-01	[ 1.2961768e+00]	 1.8758631e-01


.. parsed-literal::

      83	 1.4319109e+00	 1.2907227e-01	 1.4845483e+00	 1.3767402e-01	[ 1.2994152e+00]	 2.1631193e-01


.. parsed-literal::

      84	 1.4360765e+00	 1.2902677e-01	 1.4887972e+00	 1.3775521e-01	[ 1.2998770e+00]	 2.1428919e-01


.. parsed-literal::

      85	 1.4410169e+00	 1.2892946e-01	 1.4939625e+00	 1.3770411e-01	  1.2975820e+00 	 2.1350479e-01
      86	 1.4441916e+00	 1.2897877e-01	 1.4972863e+00	 1.3819575e-01	  1.2965752e+00 	 2.1083784e-01


.. parsed-literal::

      87	 1.4481466e+00	 1.2868414e-01	 1.5011725e+00	 1.3788108e-01	  1.2992093e+00 	 2.0880294e-01


.. parsed-literal::

      88	 1.4518948e+00	 1.2848172e-01	 1.5049257e+00	 1.3766575e-01	[ 1.3004704e+00]	 2.1149755e-01


.. parsed-literal::

      89	 1.4550913e+00	 1.2832132e-01	 1.5081304e+00	 1.3757932e-01	[ 1.3022127e+00]	 2.1894860e-01
      90	 1.4590881e+00	 1.2816757e-01	 1.5123233e+00	 1.3714653e-01	[ 1.3040160e+00]	 1.9566226e-01


.. parsed-literal::

      91	 1.4628470e+00	 1.2813358e-01	 1.5160995e+00	 1.3713734e-01	[ 1.3077212e+00]	 2.0877099e-01
      92	 1.4653602e+00	 1.2814190e-01	 1.5187329e+00	 1.3698639e-01	  1.3068156e+00 	 1.8997812e-01


.. parsed-literal::

      93	 1.4681333e+00	 1.2803136e-01	 1.5216700e+00	 1.3663853e-01	[ 1.3090131e+00]	 2.1619296e-01


.. parsed-literal::

      94	 1.4706562e+00	 1.2797009e-01	 1.5242878e+00	 1.3630005e-01	  1.3079265e+00 	 2.1330905e-01


.. parsed-literal::

      95	 1.4723325e+00	 1.2787609e-01	 1.5258797e+00	 1.3619207e-01	[ 1.3097166e+00]	 2.1050334e-01


.. parsed-literal::

      96	 1.4753480e+00	 1.2774552e-01	 1.5288210e+00	 1.3607852e-01	[ 1.3115430e+00]	 2.0770407e-01


.. parsed-literal::

      97	 1.4785241e+00	 1.2750205e-01	 1.5319332e+00	 1.3584805e-01	[ 1.3139665e+00]	 2.1607161e-01
      98	 1.4808415e+00	 1.2738406e-01	 1.5342695e+00	 1.3599945e-01	[ 1.3154286e+00]	 2.0801806e-01


.. parsed-literal::

      99	 1.4839758e+00	 1.2719286e-01	 1.5373925e+00	 1.3553898e-01	[ 1.3165368e+00]	 2.1518254e-01
     100	 1.4854275e+00	 1.2711424e-01	 1.5388828e+00	 1.3528883e-01	[ 1.3171190e+00]	 1.8894267e-01


.. parsed-literal::

     101	 1.4880413e+00	 1.2694884e-01	 1.5416595e+00	 1.3470830e-01	  1.3148856e+00 	 1.9335651e-01


.. parsed-literal::

     102	 1.4902054e+00	 1.2675387e-01	 1.5439886e+00	 1.3390245e-01	  1.3168142e+00 	 2.1020484e-01


.. parsed-literal::

     103	 1.4930831e+00	 1.2663694e-01	 1.5468084e+00	 1.3370191e-01	  1.3164374e+00 	 2.1535635e-01


.. parsed-literal::

     104	 1.4949605e+00	 1.2646952e-01	 1.5485913e+00	 1.3353718e-01	[ 1.3184990e+00]	 2.0891380e-01
     105	 1.4966913e+00	 1.2628072e-01	 1.5502597e+00	 1.3327436e-01	[ 1.3202055e+00]	 1.7674923e-01


.. parsed-literal::

     106	 1.4985583e+00	 1.2576247e-01	 1.5521080e+00	 1.3271961e-01	[ 1.3281228e+00]	 1.8627763e-01


.. parsed-literal::

     107	 1.5013480e+00	 1.2569438e-01	 1.5548534e+00	 1.3277094e-01	  1.3269837e+00 	 2.1113753e-01
     108	 1.5028332e+00	 1.2562791e-01	 1.5563810e+00	 1.3285980e-01	  1.3260125e+00 	 1.8120551e-01


.. parsed-literal::

     109	 1.5044806e+00	 1.2549924e-01	 1.5581027e+00	 1.3305678e-01	  1.3250884e+00 	 2.0545721e-01
     110	 1.5073021e+00	 1.2525054e-01	 1.5610905e+00	 1.3366603e-01	  1.3211433e+00 	 1.9570637e-01


.. parsed-literal::

     111	 1.5090599e+00	 1.2501915e-01	 1.5629645e+00	 1.3376949e-01	  1.3187695e+00 	 2.8834081e-01


.. parsed-literal::

     112	 1.5109428e+00	 1.2488639e-01	 1.5648568e+00	 1.3407224e-01	  1.3183603e+00 	 2.1153307e-01
     113	 1.5129214e+00	 1.2465906e-01	 1.5668059e+00	 1.3410902e-01	  1.3183267e+00 	 1.8469882e-01


.. parsed-literal::

     114	 1.5149955e+00	 1.2437158e-01	 1.5688581e+00	 1.3396360e-01	  1.3214692e+00 	 2.0558453e-01


.. parsed-literal::

     115	 1.5170904e+00	 1.2398106e-01	 1.5709424e+00	 1.3355290e-01	  1.3185010e+00 	 2.0551729e-01
     116	 1.5188549e+00	 1.2381869e-01	 1.5726888e+00	 1.3317485e-01	  1.3203450e+00 	 1.8124437e-01


.. parsed-literal::

     117	 1.5213559e+00	 1.2352969e-01	 1.5752459e+00	 1.3261716e-01	  1.3193578e+00 	 2.1576071e-01


.. parsed-literal::

     118	 1.5235265e+00	 1.2322826e-01	 1.5775329e+00	 1.3218039e-01	  1.3155259e+00 	 2.0712996e-01


.. parsed-literal::

     119	 1.5254918e+00	 1.2298368e-01	 1.5796358e+00	 1.3210598e-01	  1.3084308e+00 	 2.0849609e-01
     120	 1.5266651e+00	 1.2285601e-01	 1.5809005e+00	 1.3197834e-01	  1.3070574e+00 	 1.8083334e-01


.. parsed-literal::

     121	 1.5277920e+00	 1.2282397e-01	 1.5820024e+00	 1.3208469e-01	  1.3066900e+00 	 2.1666288e-01
     122	 1.5292702e+00	 1.2274060e-01	 1.5834662e+00	 1.3211239e-01	  1.3069379e+00 	 1.9387579e-01


.. parsed-literal::

     123	 1.5309743e+00	 1.2261295e-01	 1.5851793e+00	 1.3197808e-01	  1.3077864e+00 	 2.0752192e-01
     124	 1.5326458e+00	 1.2240568e-01	 1.5869104e+00	 1.3189160e-01	  1.3067392e+00 	 1.9932079e-01


.. parsed-literal::

     125	 1.5345996e+00	 1.2231185e-01	 1.5888098e+00	 1.3166467e-01	  1.3091367e+00 	 1.9739628e-01


.. parsed-literal::

     126	 1.5358947e+00	 1.2219451e-01	 1.5901093e+00	 1.3142241e-01	  1.3080648e+00 	 2.2090840e-01


.. parsed-literal::

     127	 1.5373807e+00	 1.2209735e-01	 1.5916619e+00	 1.3135125e-01	  1.3090410e+00 	 2.1750641e-01


.. parsed-literal::

     128	 1.5387898e+00	 1.2192669e-01	 1.5930916e+00	 1.3109332e-01	  1.3048342e+00 	 2.0081758e-01


.. parsed-literal::

     129	 1.5399996e+00	 1.2183817e-01	 1.5943028e+00	 1.3101277e-01	  1.3043930e+00 	 2.0986414e-01


.. parsed-literal::

     130	 1.5416339e+00	 1.2166319e-01	 1.5959940e+00	 1.3096608e-01	  1.3016326e+00 	 2.0766473e-01


.. parsed-literal::

     131	 1.5428012e+00	 1.2156761e-01	 1.5971483e+00	 1.3067638e-01	  1.3048988e+00 	 2.0937610e-01


.. parsed-literal::

     132	 1.5436276e+00	 1.2153221e-01	 1.5979445e+00	 1.3054512e-01	  1.3064514e+00 	 2.0996523e-01


.. parsed-literal::

     133	 1.5464318e+00	 1.2135904e-01	 1.6006602e+00	 1.2992517e-01	  1.3098651e+00 	 2.0718217e-01


.. parsed-literal::

     134	 1.5471803e+00	 1.2121469e-01	 1.6014106e+00	 1.2961973e-01	  1.3092489e+00 	 2.1111774e-01


.. parsed-literal::

     135	 1.5490591e+00	 1.2111258e-01	 1.6032175e+00	 1.2936716e-01	  1.3101684e+00 	 2.0614457e-01
     136	 1.5501568e+00	 1.2099598e-01	 1.6043027e+00	 1.2921823e-01	  1.3094646e+00 	 1.9059563e-01


.. parsed-literal::

     137	 1.5510430e+00	 1.2087964e-01	 1.6052112e+00	 1.2911006e-01	  1.3081394e+00 	 1.7989063e-01
     138	 1.5525843e+00	 1.2069254e-01	 1.6068379e+00	 1.2897961e-01	  1.3047703e+00 	 1.9951653e-01


.. parsed-literal::

     139	 1.5537138e+00	 1.2038335e-01	 1.6082001e+00	 1.2872471e-01	  1.2958134e+00 	 2.0687270e-01


.. parsed-literal::

     140	 1.5553919e+00	 1.2036062e-01	 1.6098327e+00	 1.2870999e-01	  1.2958355e+00 	 2.1374297e-01


.. parsed-literal::

     141	 1.5560968e+00	 1.2035620e-01	 1.6105387e+00	 1.2869609e-01	  1.2953054e+00 	 2.1091294e-01


.. parsed-literal::

     142	 1.5571483e+00	 1.2028420e-01	 1.6116269e+00	 1.2859707e-01	  1.2913679e+00 	 2.0427418e-01


.. parsed-literal::

     143	 1.5579573e+00	 1.2017041e-01	 1.6125786e+00	 1.2878587e-01	  1.2816683e+00 	 2.1824861e-01
     144	 1.5594185e+00	 1.2007081e-01	 1.6139867e+00	 1.2854289e-01	  1.2818944e+00 	 1.8082142e-01


.. parsed-literal::

     145	 1.5600648e+00	 1.1999993e-01	 1.6146210e+00	 1.2850352e-01	  1.2806848e+00 	 2.0712757e-01


.. parsed-literal::

     146	 1.5609639e+00	 1.1990833e-01	 1.6155286e+00	 1.2846574e-01	  1.2798833e+00 	 2.0772862e-01


.. parsed-literal::

     147	 1.5619399e+00	 1.1983494e-01	 1.6165824e+00	 1.2847151e-01	  1.2742220e+00 	 2.0542336e-01


.. parsed-literal::

     148	 1.5633570e+00	 1.1977832e-01	 1.6179792e+00	 1.2840281e-01	  1.2771176e+00 	 2.1950245e-01


.. parsed-literal::

     149	 1.5644708e+00	 1.1976260e-01	 1.6191002e+00	 1.2831791e-01	  1.2785071e+00 	 2.1070600e-01


.. parsed-literal::

     150	 1.5653534e+00	 1.1976394e-01	 1.6199935e+00	 1.2824881e-01	  1.2777591e+00 	 2.0898366e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.02 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f1ac4f337f0>



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


.. parsed-literal::

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
    CPU times: user 2.17 s, sys: 42 ms, total: 2.21 s
    Wall time: 688 ms


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

