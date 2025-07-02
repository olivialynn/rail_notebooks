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
       1	-3.4180043e-01	 3.2032572e-01	-3.3209882e-01	 3.2108337e-01	[-3.3370915e-01]	 4.6122193e-01


.. parsed-literal::

       2	-2.7250331e-01	 3.0992918e-01	-2.4891311e-01	 3.1160553e-01	[-2.5387799e-01]	 2.3206353e-01


.. parsed-literal::

       3	-2.3108856e-01	 2.9076323e-01	-1.9081409e-01	 2.9117785e-01	[-1.9438849e-01]	 2.8749776e-01
       4	-1.8733347e-01	 2.6652349e-01	-1.4453013e-01	 2.6576076e-01	[-1.4215587e-01]	 1.7582297e-01


.. parsed-literal::

       5	-1.0666370e-01	 2.5675748e-01	-7.4351283e-02	 2.5681152e-01	[-7.5553751e-02]	 1.9646907e-01
       6	-7.0018719e-02	 2.5158764e-01	-4.0567444e-02	 2.5129572e-01	[-3.9139816e-02]	 1.7490745e-01


.. parsed-literal::

       7	-5.3662079e-02	 2.4901946e-01	-2.9282441e-02	 2.4899962e-01	[-2.8303742e-02]	 2.0725346e-01
       8	-3.9642430e-02	 2.4667613e-01	-1.9486935e-02	 2.4664407e-01	[-1.8308807e-02]	 1.7938352e-01


.. parsed-literal::

       9	-2.8087355e-02	 2.4458602e-01	-1.0541824e-02	 2.4450105e-01	[-9.1591467e-03]	 1.9160366e-01
      10	-1.8509424e-02	 2.4288706e-01	-3.3433869e-03	 2.4176635e-01	[ 2.1308847e-03]	 1.9129419e-01


.. parsed-literal::

      11	-1.2420921e-02	 2.4176993e-01	 1.7720953e-03	 2.4090706e-01	[ 6.2081849e-03]	 2.1230841e-01
      12	-1.0197201e-02	 2.4144757e-01	 3.7384006e-03	 2.4052766e-01	[ 8.6968034e-03]	 1.8847609e-01


.. parsed-literal::

      13	-6.5119977e-03	 2.4076984e-01	 7.2142438e-03	 2.3962873e-01	[ 1.3216948e-02]	 2.0693302e-01


.. parsed-literal::

      14	 1.2094270e-01	 2.2729017e-01	 1.4274040e-01	 2.2405212e-01	[ 1.6007445e-01]	 4.3485212e-01


.. parsed-literal::

      15	 1.4319769e-01	 2.2616554e-01	 1.6629285e-01	 2.2351468e-01	[ 1.8499734e-01]	 2.1885133e-01
      16	 2.1354832e-01	 2.2111039e-01	 2.3825962e-01	 2.1852823e-01	[ 2.6090386e-01]	 2.0361662e-01


.. parsed-literal::

      17	 2.6150941e-01	 2.1402536e-01	 2.9272142e-01	 2.1053020e-01	[ 3.3452317e-01]	 1.9993258e-01


.. parsed-literal::

      18	 3.1171331e-01	 2.1493589e-01	 3.4167774e-01	 2.1179039e-01	[ 3.7486117e-01]	 2.0845032e-01


.. parsed-literal::

      19	 3.4545799e-01	 2.1239635e-01	 3.7653052e-01	 2.0980837e-01	[ 4.0673083e-01]	 2.0902038e-01


.. parsed-literal::

      20	 4.1127299e-01	 2.1165504e-01	 4.4300568e-01	 2.1068871e-01	[ 4.6175007e-01]	 2.1240616e-01


.. parsed-literal::

      21	 5.0250384e-01	 2.1200700e-01	 5.3620283e-01	 2.0790841e-01	[ 5.6292687e-01]	 2.1088624e-01
      22	 5.6132527e-01	 2.1291950e-01	 6.0042245e-01	 2.0799351e-01	[ 6.1261990e-01]	 1.9013977e-01


.. parsed-literal::

      23	 6.1111790e-01	 2.0454935e-01	 6.4932593e-01	 1.9858357e-01	[ 6.6467327e-01]	 2.1131754e-01


.. parsed-literal::

      24	 6.4826217e-01	 2.0083559e-01	 6.8533302e-01	 1.9467025e-01	[ 7.0064415e-01]	 2.0780921e-01
      25	 6.7376530e-01	 2.0170580e-01	 7.0781911e-01	 1.9672629e-01	[ 7.2062257e-01]	 1.9829798e-01


.. parsed-literal::

      26	 7.2723052e-01	 1.9582452e-01	 7.6399322e-01	 1.9086389e-01	[ 7.6940283e-01]	 2.0814824e-01


.. parsed-literal::

      27	 7.5292960e-01	 1.9307089e-01	 7.9122752e-01	 1.8813175e-01	[ 7.8964164e-01]	 2.1908164e-01


.. parsed-literal::

      28	 7.7061035e-01	 1.9192354e-01	 8.0817573e-01	 1.8651399e-01	[ 8.0842402e-01]	 2.1312952e-01


.. parsed-literal::

      29	 7.9565745e-01	 1.9102276e-01	 8.3412143e-01	 1.8601345e-01	[ 8.2878784e-01]	 2.1934915e-01


.. parsed-literal::

      30	 8.2666048e-01	 1.9228799e-01	 8.6588814e-01	 1.8838434e-01	[ 8.5520175e-01]	 2.1225166e-01


.. parsed-literal::

      31	 8.5570195e-01	 1.8897340e-01	 8.9627904e-01	 1.8615281e-01	[ 8.8069151e-01]	 2.1413183e-01


.. parsed-literal::

      32	 8.8626391e-01	 1.8494493e-01	 9.2748360e-01	 1.8306246e-01	[ 9.1241910e-01]	 2.1912622e-01


.. parsed-literal::

      33	 9.0771900e-01	 1.8323460e-01	 9.4956782e-01	 1.8254297e-01	[ 9.2442832e-01]	 2.1297455e-01


.. parsed-literal::

      34	 9.2698793e-01	 1.7979292e-01	 9.6843239e-01	 1.7915671e-01	[ 9.4802275e-01]	 2.1324444e-01


.. parsed-literal::

      35	 9.3927597e-01	 1.7834332e-01	 9.8079168e-01	 1.7846505e-01	[ 9.5913411e-01]	 2.1704173e-01


.. parsed-literal::

      36	 9.6004514e-01	 1.7489488e-01	 1.0021828e+00	 1.7580301e-01	[ 9.7646424e-01]	 2.1422076e-01


.. parsed-literal::

      37	 9.8168060e-01	 1.7121417e-01	 1.0253812e+00	 1.7521164e-01	[ 9.9298995e-01]	 2.1161842e-01


.. parsed-literal::

      38	 9.9594156e-01	 1.6791143e-01	 1.0398767e+00	 1.7032735e-01	[ 1.0108433e+00]	 2.1076298e-01
      39	 1.0049300e+00	 1.6846932e-01	 1.0485513e+00	 1.7118701e-01	[ 1.0210120e+00]	 1.6294241e-01


.. parsed-literal::

      40	 1.0137624e+00	 1.6718594e-01	 1.0577963e+00	 1.7039488e-01	[ 1.0283648e+00]	 1.8778992e-01
      41	 1.0318030e+00	 1.6534730e-01	 1.0771945e+00	 1.7020208e-01	[ 1.0386393e+00]	 1.6775751e-01


.. parsed-literal::

      42	 1.0477925e+00	 1.6245517e-01	 1.0937001e+00	 1.6805839e-01	[ 1.0575249e+00]	 1.8007135e-01


.. parsed-literal::

      43	 1.0598755e+00	 1.6156981e-01	 1.1066826e+00	 1.6813578e-01	  1.0573613e+00 	 2.0750022e-01


.. parsed-literal::

      44	 1.0696808e+00	 1.6015491e-01	 1.1163528e+00	 1.6700095e-01	[ 1.0674325e+00]	 2.1978045e-01


.. parsed-literal::

      45	 1.0830353e+00	 1.5802902e-01	 1.1295164e+00	 1.6474886e-01	[ 1.0816636e+00]	 2.1516514e-01


.. parsed-literal::

      46	 1.0938865e+00	 1.5576483e-01	 1.1404632e+00	 1.6314476e-01	[ 1.0889251e+00]	 2.1136403e-01


.. parsed-literal::

      47	 1.1028928e+00	 1.5388309e-01	 1.1498189e+00	 1.6203412e-01	[ 1.0994547e+00]	 2.0918059e-01


.. parsed-literal::

      48	 1.1172205e+00	 1.5229151e-01	 1.1638300e+00	 1.6147357e-01	[ 1.1170052e+00]	 2.1474099e-01
      49	 1.1228325e+00	 1.5142121e-01	 1.1695189e+00	 1.6127434e-01	[ 1.1223849e+00]	 1.9290876e-01


.. parsed-literal::

      50	 1.1368861e+00	 1.4939293e-01	 1.1842833e+00	 1.6210018e-01	[ 1.1336489e+00]	 2.0815849e-01


.. parsed-literal::

      51	 1.1430231e+00	 1.4798960e-01	 1.1904036e+00	 1.6114824e-01	[ 1.1347114e+00]	 2.1260142e-01


.. parsed-literal::

      52	 1.1515194e+00	 1.4729000e-01	 1.1987502e+00	 1.6043758e-01	[ 1.1409059e+00]	 2.0941949e-01


.. parsed-literal::

      53	 1.1593922e+00	 1.4624939e-01	 1.2067638e+00	 1.5921176e-01	[ 1.1439432e+00]	 2.0563412e-01


.. parsed-literal::

      54	 1.1660068e+00	 1.4489512e-01	 1.2137012e+00	 1.5835687e-01	  1.1406900e+00 	 2.0899415e-01


.. parsed-literal::

      55	 1.1756986e+00	 1.4355724e-01	 1.2239472e+00	 1.5647345e-01	[ 1.1455042e+00]	 2.0947242e-01


.. parsed-literal::

      56	 1.1848617e+00	 1.4218729e-01	 1.2333999e+00	 1.5579505e-01	[ 1.1470326e+00]	 2.0942855e-01


.. parsed-literal::

      57	 1.1919300e+00	 1.4204265e-01	 1.2405002e+00	 1.5563560e-01	[ 1.1582009e+00]	 2.1004629e-01


.. parsed-literal::

      58	 1.1988025e+00	 1.4183150e-01	 1.2476286e+00	 1.5562012e-01	[ 1.1650230e+00]	 2.0673656e-01


.. parsed-literal::

      59	 1.2088000e+00	 1.4148204e-01	 1.2581173e+00	 1.5563671e-01	[ 1.1717596e+00]	 2.0136619e-01


.. parsed-literal::

      60	 1.2166988e+00	 1.4178609e-01	 1.2663851e+00	 1.5675117e-01	[ 1.1734140e+00]	 2.0361996e-01


.. parsed-literal::

      61	 1.2233774e+00	 1.4125628e-01	 1.2730313e+00	 1.5589726e-01	[ 1.1793543e+00]	 2.0695329e-01
      62	 1.2293120e+00	 1.4069216e-01	 1.2789737e+00	 1.5496071e-01	[ 1.1860771e+00]	 1.9958591e-01


.. parsed-literal::

      63	 1.2360804e+00	 1.4019485e-01	 1.2859435e+00	 1.5442072e-01	[ 1.1959283e+00]	 1.8321681e-01


.. parsed-literal::

      64	 1.2406036e+00	 1.4036996e-01	 1.2907501e+00	 1.5477344e-01	[ 1.2062894e+00]	 2.0569634e-01
      65	 1.2466798e+00	 1.3989479e-01	 1.2965479e+00	 1.5425647e-01	[ 1.2118916e+00]	 1.9990468e-01


.. parsed-literal::

      66	 1.2500044e+00	 1.3988312e-01	 1.2998736e+00	 1.5464574e-01	[ 1.2159428e+00]	 2.0659328e-01
      67	 1.2552762e+00	 1.3965279e-01	 1.3051265e+00	 1.5443908e-01	[ 1.2212381e+00]	 1.9992304e-01


.. parsed-literal::

      68	 1.2645452e+00	 1.3917022e-01	 1.3146148e+00	 1.5449918e-01	[ 1.2233302e+00]	 1.9059467e-01


.. parsed-literal::

      69	 1.2722827e+00	 1.3915051e-01	 1.3224638e+00	 1.5377088e-01	[ 1.2293962e+00]	 2.0748663e-01


.. parsed-literal::

      70	 1.2774230e+00	 1.3881402e-01	 1.3275869e+00	 1.5315141e-01	[ 1.2316819e+00]	 2.0366430e-01
      71	 1.2862207e+00	 1.3865209e-01	 1.3367558e+00	 1.5206627e-01	  1.2314640e+00 	 1.7674732e-01


.. parsed-literal::

      72	 1.2940424e+00	 1.3788606e-01	 1.3445930e+00	 1.5126226e-01	[ 1.2396976e+00]	 1.9204664e-01
      73	 1.2989182e+00	 1.3779400e-01	 1.3495335e+00	 1.5111187e-01	[ 1.2444909e+00]	 1.9820833e-01


.. parsed-literal::

      74	 1.3091731e+00	 1.3764308e-01	 1.3602973e+00	 1.5053782e-01	[ 1.2530638e+00]	 2.0457268e-01


.. parsed-literal::

      75	 1.3120185e+00	 1.3747677e-01	 1.3636412e+00	 1.5003807e-01	[ 1.2553086e+00]	 2.0813632e-01


.. parsed-literal::

      76	 1.3184910e+00	 1.3699318e-01	 1.3697809e+00	 1.4977589e-01	[ 1.2601836e+00]	 2.0682883e-01
      77	 1.3237631e+00	 1.3654946e-01	 1.3750602e+00	 1.4959599e-01	[ 1.2631962e+00]	 1.8576694e-01


.. parsed-literal::

      78	 1.3299733e+00	 1.3584721e-01	 1.3814586e+00	 1.4922020e-01	[ 1.2647278e+00]	 2.0367718e-01


.. parsed-literal::

      79	 1.3385987e+00	 1.3516643e-01	 1.3901887e+00	 1.4922991e-01	[ 1.2747712e+00]	 2.0611572e-01


.. parsed-literal::

      80	 1.3413645e+00	 1.3517529e-01	 1.3934318e+00	 1.4843450e-01	[ 1.2784663e+00]	 2.0813870e-01
      81	 1.3489536e+00	 1.3475591e-01	 1.4006980e+00	 1.4835008e-01	[ 1.2871610e+00]	 1.7180252e-01


.. parsed-literal::

      82	 1.3521225e+00	 1.3456172e-01	 1.4039450e+00	 1.4811090e-01	[ 1.2903522e+00]	 1.9873714e-01


.. parsed-literal::

      83	 1.3566475e+00	 1.3426401e-01	 1.4086659e+00	 1.4741897e-01	[ 1.2930155e+00]	 2.0954680e-01


.. parsed-literal::

      84	 1.3629974e+00	 1.3399536e-01	 1.4152233e+00	 1.4689351e-01	[ 1.2994627e+00]	 2.0657206e-01


.. parsed-literal::

      85	 1.3677463e+00	 1.3359794e-01	 1.4203576e+00	 1.4614669e-01	[ 1.2995099e+00]	 2.1019077e-01


.. parsed-literal::

      86	 1.3726160e+00	 1.3345972e-01	 1.4250608e+00	 1.4622826e-01	[ 1.3032385e+00]	 2.1275687e-01


.. parsed-literal::

      87	 1.3754918e+00	 1.3319567e-01	 1.4279964e+00	 1.4632246e-01	  1.3028172e+00 	 2.1262121e-01


.. parsed-literal::

      88	 1.3790194e+00	 1.3259025e-01	 1.4316134e+00	 1.4623726e-01	  1.2997179e+00 	 2.0657706e-01
      89	 1.3842227e+00	 1.3226162e-01	 1.4370537e+00	 1.4627114e-01	  1.2932671e+00 	 1.9587111e-01


.. parsed-literal::

      90	 1.3888825e+00	 1.3193007e-01	 1.4418055e+00	 1.4597511e-01	  1.2917967e+00 	 2.0439816e-01


.. parsed-literal::

      91	 1.3932365e+00	 1.3187476e-01	 1.4461147e+00	 1.4602207e-01	  1.2969881e+00 	 2.0982885e-01
      92	 1.3973482e+00	 1.3208216e-01	 1.4502084e+00	 1.4586286e-01	  1.3021876e+00 	 2.0219827e-01


.. parsed-literal::

      93	 1.4014934e+00	 1.3174312e-01	 1.4542778e+00	 1.4568151e-01	[ 1.3075000e+00]	 1.8595099e-01


.. parsed-literal::

      94	 1.4043407e+00	 1.3159394e-01	 1.4571615e+00	 1.4530962e-01	[ 1.3136790e+00]	 2.1048355e-01


.. parsed-literal::

      95	 1.4076259e+00	 1.3132999e-01	 1.4603665e+00	 1.4514551e-01	[ 1.3138056e+00]	 2.0814013e-01
      96	 1.4110840e+00	 1.3109026e-01	 1.4638515e+00	 1.4502682e-01	  1.3117080e+00 	 1.7638350e-01


.. parsed-literal::

      97	 1.4146129e+00	 1.3101907e-01	 1.4674275e+00	 1.4508706e-01	  1.3119783e+00 	 2.1925235e-01


.. parsed-literal::

      98	 1.4166068e+00	 1.3116837e-01	 1.4695125e+00	 1.4558953e-01	  1.3112103e+00 	 2.1915221e-01
      99	 1.4204596e+00	 1.3109662e-01	 1.4732852e+00	 1.4557243e-01	[ 1.3166903e+00]	 1.9574809e-01


.. parsed-literal::

     100	 1.4223385e+00	 1.3101443e-01	 1.4751557e+00	 1.4562573e-01	[ 1.3197120e+00]	 2.1734452e-01


.. parsed-literal::

     101	 1.4250572e+00	 1.3091221e-01	 1.4779719e+00	 1.4577050e-01	[ 1.3221958e+00]	 2.1361804e-01


.. parsed-literal::

     102	 1.4287349e+00	 1.3046485e-01	 1.4818705e+00	 1.4590839e-01	  1.3213776e+00 	 2.1737885e-01


.. parsed-literal::

     103	 1.4318147e+00	 1.3046091e-01	 1.4852968e+00	 1.4616537e-01	  1.3200921e+00 	 2.0630312e-01


.. parsed-literal::

     104	 1.4347214e+00	 1.3011006e-01	 1.4881260e+00	 1.4599139e-01	  1.3190618e+00 	 2.1469140e-01
     105	 1.4368948e+00	 1.2980047e-01	 1.4902710e+00	 1.4593526e-01	  1.3180323e+00 	 1.9515753e-01


.. parsed-literal::

     106	 1.4396528e+00	 1.2953352e-01	 1.4930747e+00	 1.4590441e-01	  1.3162558e+00 	 2.0900798e-01


.. parsed-literal::

     107	 1.4429336e+00	 1.2919555e-01	 1.4964201e+00	 1.4613539e-01	  1.3161813e+00 	 2.0422530e-01


.. parsed-literal::

     108	 1.4454009e+00	 1.2921978e-01	 1.4988515e+00	 1.4618492e-01	  1.3185047e+00 	 2.1235347e-01
     109	 1.4477907e+00	 1.2925648e-01	 1.5012816e+00	 1.4621714e-01	[ 1.3228932e+00]	 1.8101549e-01


.. parsed-literal::

     110	 1.4509573e+00	 1.2927034e-01	 1.5045405e+00	 1.4611683e-01	[ 1.3245004e+00]	 2.1537971e-01
     111	 1.4536595e+00	 1.2922351e-01	 1.5073988e+00	 1.4581584e-01	[ 1.3327756e+00]	 1.9612098e-01


.. parsed-literal::

     112	 1.4561682e+00	 1.2916169e-01	 1.5098074e+00	 1.4549811e-01	[ 1.3331808e+00]	 2.0455003e-01


.. parsed-literal::

     113	 1.4576869e+00	 1.2899949e-01	 1.5113374e+00	 1.4551918e-01	  1.3303879e+00 	 2.0182848e-01


.. parsed-literal::

     114	 1.4596402e+00	 1.2874536e-01	 1.5133139e+00	 1.4546456e-01	  1.3284549e+00 	 2.1275473e-01


.. parsed-literal::

     115	 1.4619171e+00	 1.2863724e-01	 1.5156927e+00	 1.4533622e-01	  1.3310207e+00 	 2.0594382e-01


.. parsed-literal::

     116	 1.4640612e+00	 1.2844541e-01	 1.5178473e+00	 1.4555148e-01	  1.3312093e+00 	 2.0974970e-01
     117	 1.4657503e+00	 1.2847513e-01	 1.5195303e+00	 1.4563659e-01	[ 1.3333666e+00]	 1.9795489e-01


.. parsed-literal::

     118	 1.4688031e+00	 1.2853566e-01	 1.5226346e+00	 1.4581454e-01	[ 1.3345281e+00]	 1.7927694e-01


.. parsed-literal::

     119	 1.4697662e+00	 1.2833060e-01	 1.5237569e+00	 1.4578216e-01	  1.3290150e+00 	 2.1952558e-01


.. parsed-literal::

     120	 1.4722566e+00	 1.2834942e-01	 1.5261356e+00	 1.4570545e-01	  1.3313263e+00 	 2.1728563e-01


.. parsed-literal::

     121	 1.4735332e+00	 1.2829520e-01	 1.5274023e+00	 1.4556598e-01	  1.3311327e+00 	 2.1260691e-01
     122	 1.4750559e+00	 1.2818340e-01	 1.5289352e+00	 1.4531266e-01	  1.3296298e+00 	 1.9804406e-01


.. parsed-literal::

     123	 1.4775354e+00	 1.2793815e-01	 1.5314483e+00	 1.4494903e-01	  1.3284658e+00 	 2.1507573e-01


.. parsed-literal::

     124	 1.4791413e+00	 1.2777915e-01	 1.5330823e+00	 1.4459021e-01	  1.3271376e+00 	 3.1232548e-01
     125	 1.4810532e+00	 1.2759289e-01	 1.5349786e+00	 1.4431309e-01	  1.3288859e+00 	 1.8978190e-01


.. parsed-literal::

     126	 1.4825272e+00	 1.2740733e-01	 1.5364533e+00	 1.4417932e-01	  1.3286954e+00 	 2.1283865e-01


.. parsed-literal::

     127	 1.4841234e+00	 1.2725716e-01	 1.5380319e+00	 1.4392290e-01	  1.3284988e+00 	 2.2007585e-01


.. parsed-literal::

     128	 1.4857671e+00	 1.2721670e-01	 1.5396646e+00	 1.4363554e-01	  1.3270838e+00 	 2.0851326e-01


.. parsed-literal::

     129	 1.4873269e+00	 1.2707415e-01	 1.5412409e+00	 1.4344782e-01	  1.3236699e+00 	 2.2789907e-01


.. parsed-literal::

     130	 1.4886139e+00	 1.2705853e-01	 1.5425575e+00	 1.4331485e-01	  1.3214875e+00 	 2.0800972e-01


.. parsed-literal::

     131	 1.4902714e+00	 1.2697776e-01	 1.5442745e+00	 1.4315574e-01	  1.3197795e+00 	 2.0883989e-01


.. parsed-literal::

     132	 1.4917040e+00	 1.2689076e-01	 1.5458243e+00	 1.4318302e-01	  1.3160866e+00 	 2.0844030e-01


.. parsed-literal::

     133	 1.4931067e+00	 1.2687200e-01	 1.5472370e+00	 1.4307255e-01	  1.3167570e+00 	 2.1245289e-01
     134	 1.4941790e+00	 1.2686547e-01	 1.5483121e+00	 1.4302364e-01	  1.3177811e+00 	 2.0261955e-01


.. parsed-literal::

     135	 1.4957794e+00	 1.2688967e-01	 1.5499482e+00	 1.4293284e-01	  1.3171601e+00 	 1.9949794e-01
     136	 1.4972522e+00	 1.2699677e-01	 1.5515713e+00	 1.4283082e-01	  1.3133986e+00 	 1.7414355e-01


.. parsed-literal::

     137	 1.4992451e+00	 1.2701836e-01	 1.5535064e+00	 1.4279881e-01	  1.3139297e+00 	 2.1117425e-01


.. parsed-literal::

     138	 1.5003813e+00	 1.2694879e-01	 1.5546179e+00	 1.4278586e-01	  1.3140084e+00 	 2.1061897e-01


.. parsed-literal::

     139	 1.5018528e+00	 1.2690452e-01	 1.5561271e+00	 1.4279924e-01	  1.3124379e+00 	 2.0234847e-01


.. parsed-literal::

     140	 1.5024628e+00	 1.2680374e-01	 1.5569649e+00	 1.4276354e-01	  1.3070275e+00 	 2.1529460e-01
     141	 1.5045925e+00	 1.2678883e-01	 1.5590327e+00	 1.4284550e-01	  1.3079378e+00 	 1.9324636e-01


.. parsed-literal::

     142	 1.5055262e+00	 1.2683851e-01	 1.5600049e+00	 1.4289531e-01	  1.3065436e+00 	 1.8597150e-01


.. parsed-literal::

     143	 1.5069363e+00	 1.2684578e-01	 1.5615302e+00	 1.4296090e-01	  1.3020765e+00 	 2.1546412e-01
     144	 1.5075701e+00	 1.2708679e-01	 1.5623059e+00	 1.4308508e-01	  1.2981530e+00 	 2.0372057e-01


.. parsed-literal::

     145	 1.5091909e+00	 1.2689138e-01	 1.5638956e+00	 1.4299611e-01	  1.2980038e+00 	 2.1632886e-01


.. parsed-literal::

     146	 1.5099650e+00	 1.2683329e-01	 1.5646695e+00	 1.4292697e-01	  1.2968738e+00 	 2.0447731e-01


.. parsed-literal::

     147	 1.5108056e+00	 1.2679510e-01	 1.5654929e+00	 1.4283275e-01	  1.2961797e+00 	 2.0086956e-01


.. parsed-literal::

     148	 1.5116267e+00	 1.2678057e-01	 1.5663155e+00	 1.4255755e-01	  1.2961587e+00 	 2.0863724e-01


.. parsed-literal::

     149	 1.5130114e+00	 1.2679588e-01	 1.5676335e+00	 1.4250199e-01	  1.2966100e+00 	 2.0796561e-01


.. parsed-literal::

     150	 1.5140374e+00	 1.2681468e-01	 1.5686366e+00	 1.4243183e-01	  1.2972086e+00 	 2.1365118e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.13 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f564ca31960>



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
    CPU times: user 1.82 s, sys: 55.9 ms, total: 1.88 s
    Wall time: 594 ms


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

