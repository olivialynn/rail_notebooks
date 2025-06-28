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
       1	-3.5443094e-01	 3.2398946e-01	-3.4479409e-01	 3.0657101e-01	[-3.1319083e-01]	 4.6187091e-01


.. parsed-literal::

       2	-2.8412207e-01	 3.1365473e-01	-2.6052758e-01	 2.9688513e-01	[-2.1145194e-01]	 2.4052691e-01


.. parsed-literal::

       3	-2.3867816e-01	 2.9190412e-01	-1.9600019e-01	 2.7671572e-01	[-1.3546646e-01]	 2.9491448e-01
       4	-2.0173128e-01	 2.6826137e-01	-1.6077812e-01	 2.5557051e-01	[-9.1075016e-02]	 1.8097806e-01


.. parsed-literal::

       5	-1.0743626e-01	 2.5832360e-01	-7.3854572e-02	 2.4792565e-01	[-2.6714785e-02]	 1.9011402e-01


.. parsed-literal::

       6	-7.6339914e-02	 2.5364364e-01	-4.7395339e-02	 2.4419131e-01	[-1.3699225e-02]	 2.1439195e-01


.. parsed-literal::

       7	-5.7509269e-02	 2.5042606e-01	-3.4582464e-02	 2.4208571e-01	[-2.2125097e-03]	 2.1224117e-01


.. parsed-literal::

       8	-4.6571475e-02	 2.4856932e-01	-2.6920878e-02	 2.4076865e-01	[ 4.1416953e-03]	 2.1166444e-01


.. parsed-literal::

       9	-3.4970149e-02	 2.4641212e-01	-1.7743357e-02	 2.3912978e-01	[ 1.1700361e-02]	 2.0129871e-01


.. parsed-literal::

      10	-2.2939622e-02	 2.4396792e-01	-7.2249730e-03	 2.3635569e-01	[ 2.4078103e-02]	 2.1152067e-01
      11	-1.9643665e-02	 2.4380937e-01	-5.4012472e-03	 2.3617643e-01	[ 2.5003508e-02]	 1.9060326e-01


.. parsed-literal::

      12	-1.4751055e-02	 2.4271346e-01	-8.7969242e-04	 2.3519523e-01	[ 3.0178244e-02]	 2.1720815e-01


.. parsed-literal::

      13	-1.1886429e-02	 2.4209632e-01	 1.9142180e-03	 2.3449959e-01	[ 3.2429986e-02]	 2.1193099e-01


.. parsed-literal::

      14	-7.3948024e-03	 2.4109266e-01	 6.9602445e-03	 2.3318815e-01	[ 4.0854360e-02]	 2.0923710e-01


.. parsed-literal::

      15	 8.7438290e-02	 2.2763861e-01	 1.0889697e-01	 2.2396924e-01	[ 1.1904538e-01]	 4.3083143e-01


.. parsed-literal::

      16	 1.8960563e-01	 2.2173366e-01	 2.1421318e-01	 2.1831915e-01	[ 2.2998946e-01]	 2.1369290e-01


.. parsed-literal::

      17	 2.4907629e-01	 2.1841851e-01	 2.7700408e-01	 2.1373317e-01	[ 3.0216811e-01]	 3.2736802e-01
      18	 3.0669462e-01	 2.1714572e-01	 3.3517506e-01	 2.1391528e-01	[ 3.5529165e-01]	 2.0129299e-01


.. parsed-literal::

      19	 3.5408236e-01	 2.1451873e-01	 3.8387342e-01	 2.1008115e-01	[ 4.0607193e-01]	 2.0609951e-01
      20	 4.1187328e-01	 2.1350589e-01	 4.4391102e-01	 2.0684352e-01	[ 4.7415688e-01]	 1.8153667e-01


.. parsed-literal::

      21	 4.6774484e-01	 2.1670375e-01	 5.0182879e-01	 2.1015912e-01	[ 5.2989302e-01]	 2.0178533e-01


.. parsed-literal::

      22	 5.2232632e-01	 2.1416459e-01	 5.5821962e-01	 2.0806594e-01	[ 5.8394283e-01]	 2.0860648e-01


.. parsed-literal::

      23	 5.7663915e-01	 2.0994268e-01	 6.1404098e-01	 2.0478526e-01	[ 6.2814092e-01]	 2.0716500e-01
      24	 6.1172199e-01	 2.0794819e-01	 6.5087590e-01	 2.0248580e-01	[ 6.3979665e-01]	 1.9966865e-01


.. parsed-literal::

      25	 6.3587551e-01	 2.0625872e-01	 6.7424142e-01	 2.0012382e-01	[ 6.6445607e-01]	 2.0063138e-01


.. parsed-literal::

      26	 6.5389206e-01	 2.0490096e-01	 6.9106280e-01	 1.9914403e-01	[ 6.8343446e-01]	 2.1501875e-01


.. parsed-literal::

      27	 6.7491222e-01	 2.0496023e-01	 7.1164806e-01	 1.9983726e-01	[ 7.0352344e-01]	 2.0709300e-01


.. parsed-literal::

      28	 6.9203144e-01	 2.0537204e-01	 7.2812535e-01	 1.9853322e-01	[ 7.3204862e-01]	 2.2194529e-01


.. parsed-literal::

      29	 7.2888127e-01	 2.0470270e-01	 7.6626025e-01	 1.9987008e-01	[ 7.6034157e-01]	 2.1088552e-01


.. parsed-literal::

      30	 7.5092222e-01	 2.0562233e-01	 7.8951896e-01	 2.0084818e-01	[ 7.7275865e-01]	 2.2281909e-01
      31	 7.7347235e-01	 2.0665485e-01	 8.1243713e-01	 2.0058585e-01	[ 7.9950553e-01]	 2.0069194e-01


.. parsed-literal::

      32	 7.9901858e-01	 2.1010242e-01	 8.3847302e-01	 2.0134340e-01	[ 8.2974854e-01]	 1.8083167e-01
      33	 8.1421700e-01	 2.1242665e-01	 8.5534187e-01	 2.0120776e-01	[ 8.4387160e-01]	 1.9875956e-01


.. parsed-literal::

      34	 8.3113008e-01	 2.1111873e-01	 8.7280697e-01	 2.0025167e-01	[ 8.5719677e-01]	 2.0859885e-01


.. parsed-literal::

      35	 8.4752330e-01	 2.1016353e-01	 8.8949901e-01	 1.9728267e-01	[ 8.8571540e-01]	 2.0513821e-01
      36	 8.6741783e-01	 2.0946506e-01	 9.0994575e-01	 1.9634680e-01	[ 8.9728401e-01]	 1.9976759e-01


.. parsed-literal::

      37	 8.9219489e-01	 2.0893922e-01	 9.3531374e-01	 1.9486492e-01	[ 9.2167155e-01]	 2.1611977e-01


.. parsed-literal::

      38	 9.0742915e-01	 2.0685161e-01	 9.5095197e-01	 1.9327458e-01	[ 9.2805244e-01]	 2.0375252e-01


.. parsed-literal::

      39	 9.2702412e-01	 2.0550282e-01	 9.7080364e-01	 1.9263603e-01	[ 9.4418931e-01]	 2.0614338e-01


.. parsed-literal::

      40	 9.4275241e-01	 2.0426179e-01	 9.8651292e-01	 1.9215624e-01	[ 9.5908221e-01]	 2.1187615e-01


.. parsed-literal::

      41	 9.5398062e-01	 2.0341265e-01	 9.9815976e-01	 1.9227554e-01	[ 9.6927682e-01]	 2.1872258e-01


.. parsed-literal::

      42	 9.6447729e-01	 2.0304145e-01	 1.0090686e+00	 1.9241979e-01	[ 9.7673821e-01]	 2.1071005e-01


.. parsed-literal::

      43	 9.7662546e-01	 2.0211298e-01	 1.0220473e+00	 1.9165732e-01	[ 9.8548346e-01]	 2.1258140e-01
      44	 9.8825531e-01	 2.0115428e-01	 1.0339577e+00	 1.9031909e-01	[ 9.9124679e-01]	 1.7912459e-01


.. parsed-literal::

      45	 1.0025624e+00	 1.9946363e-01	 1.0489666e+00	 1.8780593e-01	[ 1.0018203e+00]	 2.0403433e-01
      46	 1.0169085e+00	 1.9795880e-01	 1.0640691e+00	 1.8534165e-01	[ 1.0149028e+00]	 1.8267369e-01


.. parsed-literal::

      47	 1.0277973e+00	 1.9609547e-01	 1.0749127e+00	 1.8432553e-01	[ 1.0271161e+00]	 2.1265125e-01


.. parsed-literal::

      48	 1.0365878e+00	 1.9471564e-01	 1.0838743e+00	 1.8313216e-01	[ 1.0395427e+00]	 2.1351838e-01


.. parsed-literal::

      49	 1.0478733e+00	 1.9281198e-01	 1.0952004e+00	 1.8105885e-01	[ 1.0567132e+00]	 2.0686030e-01


.. parsed-literal::

      50	 1.0595385e+00	 1.9147768e-01	 1.1067006e+00	 1.7893544e-01	[ 1.0761811e+00]	 2.0790124e-01


.. parsed-literal::

      51	 1.0715130e+00	 1.8870462e-01	 1.1188491e+00	 1.7635260e-01	[ 1.0887350e+00]	 2.1598554e-01
      52	 1.0849546e+00	 1.8538643e-01	 1.1324944e+00	 1.7359365e-01	[ 1.1017368e+00]	 1.7639184e-01


.. parsed-literal::

      53	 1.0990448e+00	 1.8015270e-01	 1.1467748e+00	 1.7076663e-01	[ 1.1083029e+00]	 2.0096684e-01
      54	 1.1125735e+00	 1.7863210e-01	 1.1605356e+00	 1.6946910e-01	[ 1.1141623e+00]	 1.8750620e-01


.. parsed-literal::

      55	 1.1225708e+00	 1.7798290e-01	 1.1706721e+00	 1.6911140e-01	[ 1.1233744e+00]	 2.0908856e-01
      56	 1.1351436e+00	 1.7666403e-01	 1.1839327e+00	 1.6859710e-01	[ 1.1278032e+00]	 1.8269491e-01


.. parsed-literal::

      57	 1.1485879e+00	 1.7304051e-01	 1.1975640e+00	 1.6614058e-01	[ 1.1379058e+00]	 1.9980812e-01


.. parsed-literal::

      58	 1.1641985e+00	 1.6984977e-01	 1.2140940e+00	 1.6349601e-01	  1.1364449e+00 	 2.0895600e-01


.. parsed-literal::

      59	 1.1746556e+00	 1.6580446e-01	 1.2256333e+00	 1.6027231e-01	[ 1.1400590e+00]	 2.1443486e-01
      60	 1.1872359e+00	 1.6584926e-01	 1.2378734e+00	 1.5854098e-01	[ 1.1507297e+00]	 1.8058181e-01


.. parsed-literal::

      61	 1.1965471e+00	 1.6468403e-01	 1.2469730e+00	 1.5688061e-01	[ 1.1628961e+00]	 1.9255424e-01


.. parsed-literal::

      62	 1.2088577e+00	 1.6326440e-01	 1.2592899e+00	 1.5436428e-01	[ 1.1813869e+00]	 2.0629144e-01
      63	 1.2197635e+00	 1.5980676e-01	 1.2703844e+00	 1.5154609e-01	[ 1.1914628e+00]	 1.9788766e-01


.. parsed-literal::

      64	 1.2312071e+00	 1.5816122e-01	 1.2818391e+00	 1.5062783e-01	[ 1.1962539e+00]	 2.1411872e-01


.. parsed-literal::

      65	 1.2400595e+00	 1.5679860e-01	 1.2908879e+00	 1.5020901e-01	[ 1.1971092e+00]	 2.0917916e-01
      66	 1.2502642e+00	 1.5517210e-01	 1.3013733e+00	 1.4900370e-01	[ 1.2007655e+00]	 1.7459917e-01


.. parsed-literal::

      67	 1.2600039e+00	 1.5384364e-01	 1.3115064e+00	 1.4836591e-01	[ 1.2056508e+00]	 2.0407677e-01


.. parsed-literal::

      68	 1.2682757e+00	 1.5306428e-01	 1.3197401e+00	 1.4715459e-01	[ 1.2154323e+00]	 2.2013950e-01
      69	 1.2750953e+00	 1.5246725e-01	 1.3265233e+00	 1.4665762e-01	[ 1.2272465e+00]	 2.0774508e-01


.. parsed-literal::

      70	 1.2836481e+00	 1.5174446e-01	 1.3354047e+00	 1.4603204e-01	[ 1.2334546e+00]	 2.0914388e-01
      71	 1.2898341e+00	 1.4974552e-01	 1.3420699e+00	 1.4575121e-01	[ 1.2412711e+00]	 1.8400240e-01


.. parsed-literal::

      72	 1.2964881e+00	 1.4925504e-01	 1.3486121e+00	 1.4555172e-01	[ 1.2463207e+00]	 1.9910359e-01


.. parsed-literal::

      73	 1.3021472e+00	 1.4857910e-01	 1.3543087e+00	 1.4527458e-01	[ 1.2493228e+00]	 2.0860147e-01


.. parsed-literal::

      74	 1.3082550e+00	 1.4756476e-01	 1.3605963e+00	 1.4477184e-01	[ 1.2530422e+00]	 2.0936179e-01
      75	 1.3160225e+00	 1.4566249e-01	 1.3684555e+00	 1.4219267e-01	[ 1.2562434e+00]	 1.9815850e-01


.. parsed-literal::

      76	 1.3243425e+00	 1.4456823e-01	 1.3769093e+00	 1.4194563e-01	[ 1.2646328e+00]	 2.0750713e-01


.. parsed-literal::

      77	 1.3288986e+00	 1.4462703e-01	 1.3813620e+00	 1.4159675e-01	[ 1.2687189e+00]	 2.1336842e-01


.. parsed-literal::

      78	 1.3367368e+00	 1.4351111e-01	 1.3896491e+00	 1.4007821e-01	  1.2669467e+00 	 2.0292497e-01


.. parsed-literal::

      79	 1.3414295e+00	 1.4319354e-01	 1.3944823e+00	 1.3963300e-01	  1.2629835e+00 	 2.1222162e-01


.. parsed-literal::

      80	 1.3454795e+00	 1.4282430e-01	 1.3984079e+00	 1.3941618e-01	  1.2657230e+00 	 2.1555305e-01
      81	 1.3517025e+00	 1.4189205e-01	 1.4046895e+00	 1.3886418e-01	  1.2660163e+00 	 1.8759227e-01


.. parsed-literal::

      82	 1.3563403e+00	 1.4153391e-01	 1.4093931e+00	 1.3845452e-01	[ 1.2689748e+00]	 2.0601010e-01


.. parsed-literal::

      83	 1.3645788e+00	 1.4023433e-01	 1.4180548e+00	 1.3733012e-01	[ 1.2701774e+00]	 2.0931363e-01


.. parsed-literal::

      84	 1.3717379e+00	 1.4057559e-01	 1.4252333e+00	 1.3693581e-01	[ 1.2833568e+00]	 2.1094012e-01
      85	 1.3755684e+00	 1.4023884e-01	 1.4289490e+00	 1.3655142e-01	[ 1.2880175e+00]	 1.9968724e-01


.. parsed-literal::

      86	 1.3825161e+00	 1.3943144e-01	 1.4359249e+00	 1.3523340e-01	[ 1.2923118e+00]	 2.0252037e-01


.. parsed-literal::

      87	 1.3869946e+00	 1.3837773e-01	 1.4404965e+00	 1.3429638e-01	  1.2906119e+00 	 2.0265031e-01


.. parsed-literal::

      88	 1.3923591e+00	 1.3778285e-01	 1.4457577e+00	 1.3402245e-01	[ 1.2923877e+00]	 2.1162152e-01


.. parsed-literal::

      89	 1.3986243e+00	 1.3689445e-01	 1.4520326e+00	 1.3382835e-01	  1.2899022e+00 	 2.0995164e-01
      90	 1.4021080e+00	 1.3663204e-01	 1.4555759e+00	 1.3383141e-01	  1.2911663e+00 	 1.8892097e-01


.. parsed-literal::

      91	 1.4071256e+00	 1.3621415e-01	 1.4607094e+00	 1.3379674e-01	  1.2921760e+00 	 1.9472241e-01


.. parsed-literal::

      92	 1.4114415e+00	 1.3549519e-01	 1.4651334e+00	 1.3337445e-01	  1.2917592e+00 	 2.1203828e-01
      93	 1.4160545e+00	 1.3538167e-01	 1.4698092e+00	 1.3222153e-01	[ 1.2928301e+00]	 1.8674278e-01


.. parsed-literal::

      94	 1.4199576e+00	 1.3453256e-01	 1.4737320e+00	 1.3180450e-01	[ 1.2949213e+00]	 1.8299890e-01


.. parsed-literal::

      95	 1.4234466e+00	 1.3455796e-01	 1.4771136e+00	 1.3131561e-01	[ 1.2992868e+00]	 2.0986795e-01


.. parsed-literal::

      96	 1.4281463e+00	 1.3427604e-01	 1.4818507e+00	 1.3053569e-01	[ 1.2993422e+00]	 2.1623206e-01


.. parsed-literal::

      97	 1.4325202e+00	 1.3437212e-01	 1.4862903e+00	 1.3019061e-01	[ 1.3022203e+00]	 2.0746565e-01
      98	 1.4370447e+00	 1.3399303e-01	 1.4907524e+00	 1.2999679e-01	[ 1.3031375e+00]	 1.9953585e-01


.. parsed-literal::

      99	 1.4425408e+00	 1.3334538e-01	 1.4963500e+00	 1.3002130e-01	[ 1.3042032e+00]	 1.9841385e-01


.. parsed-literal::

     100	 1.4448001e+00	 1.3308021e-01	 1.4987332e+00	 1.3045550e-01	  1.2989265e+00 	 2.1823096e-01


.. parsed-literal::

     101	 1.4482500e+00	 1.3310885e-01	 1.5021556e+00	 1.3036357e-01	[ 1.3042399e+00]	 2.0769858e-01


.. parsed-literal::

     102	 1.4518805e+00	 1.3301446e-01	 1.5058972e+00	 1.3026812e-01	[ 1.3088726e+00]	 2.1264243e-01
     103	 1.4544033e+00	 1.3278110e-01	 1.5084284e+00	 1.3022309e-01	[ 1.3100979e+00]	 1.7611384e-01


.. parsed-literal::

     104	 1.4570415e+00	 1.3237772e-01	 1.5113420e+00	 1.3053210e-01	[ 1.3153378e+00]	 2.0404077e-01


.. parsed-literal::

     105	 1.4613379e+00	 1.3208152e-01	 1.5154325e+00	 1.3025331e-01	  1.3109456e+00 	 2.1677995e-01
     106	 1.4627834e+00	 1.3204381e-01	 1.5167950e+00	 1.2999977e-01	  1.3124156e+00 	 1.8949819e-01


.. parsed-literal::

     107	 1.4660614e+00	 1.3193356e-01	 1.5200685e+00	 1.2943814e-01	  1.3120952e+00 	 1.9093823e-01
     108	 1.4695406e+00	 1.3171312e-01	 1.5235813e+00	 1.2853440e-01	  1.3130249e+00 	 2.0060301e-01


.. parsed-literal::

     109	 1.4726594e+00	 1.3160362e-01	 1.5267715e+00	 1.2788838e-01	  1.3114740e+00 	 2.1553016e-01


.. parsed-literal::

     110	 1.4756307e+00	 1.3135810e-01	 1.5296930e+00	 1.2788557e-01	[ 1.3164567e+00]	 2.1771193e-01


.. parsed-literal::

     111	 1.4786316e+00	 1.3112408e-01	 1.5327278e+00	 1.2763804e-01	[ 1.3180142e+00]	 2.1403694e-01
     112	 1.4806515e+00	 1.3090871e-01	 1.5348630e+00	 1.2773958e-01	[ 1.3248551e+00]	 1.9996619e-01


.. parsed-literal::

     113	 1.4833937e+00	 1.3079487e-01	 1.5376151e+00	 1.2734394e-01	  1.3235200e+00 	 2.2640562e-01


.. parsed-literal::

     114	 1.4857474e+00	 1.3072225e-01	 1.5400912e+00	 1.2682426e-01	  1.3198025e+00 	 2.0939040e-01


.. parsed-literal::

     115	 1.4872581e+00	 1.3071856e-01	 1.5416586e+00	 1.2649203e-01	  1.3206257e+00 	 2.0275116e-01


.. parsed-literal::

     116	 1.4900456e+00	 1.3061000e-01	 1.5446637e+00	 1.2601890e-01	  1.3181836e+00 	 2.1042991e-01


.. parsed-literal::

     117	 1.4927331e+00	 1.3077843e-01	 1.5473296e+00	 1.2562036e-01	[ 1.3279444e+00]	 2.1947193e-01


.. parsed-literal::

     118	 1.4942246e+00	 1.3078250e-01	 1.5487534e+00	 1.2554889e-01	[ 1.3292525e+00]	 2.1349478e-01


.. parsed-literal::

     119	 1.4962963e+00	 1.3075081e-01	 1.5508235e+00	 1.2555189e-01	[ 1.3311622e+00]	 2.0857692e-01


.. parsed-literal::

     120	 1.4983691e+00	 1.3075910e-01	 1.5530085e+00	 1.2556501e-01	  1.3292890e+00 	 2.1570158e-01


.. parsed-literal::

     121	 1.5004889e+00	 1.3063762e-01	 1.5552604e+00	 1.2569536e-01	  1.3296187e+00 	 2.1290898e-01


.. parsed-literal::

     122	 1.5023856e+00	 1.3057357e-01	 1.5571783e+00	 1.2559956e-01	  1.3273402e+00 	 2.0820069e-01
     123	 1.5042815e+00	 1.3042693e-01	 1.5590862e+00	 1.2543805e-01	  1.3261073e+00 	 1.9308615e-01


.. parsed-literal::

     124	 1.5066410e+00	 1.3027725e-01	 1.5614881e+00	 1.2507608e-01	  1.3267821e+00 	 2.1114802e-01


.. parsed-literal::

     125	 1.5086484e+00	 1.3003247e-01	 1.5636341e+00	 1.2491782e-01	  1.3259325e+00 	 2.0722246e-01
     126	 1.5107799e+00	 1.3007318e-01	 1.5656337e+00	 1.2446454e-01	  1.3306590e+00 	 2.0476508e-01


.. parsed-literal::

     127	 1.5121600e+00	 1.3005481e-01	 1.5669354e+00	 1.2431598e-01	[ 1.3346321e+00]	 1.9477034e-01


.. parsed-literal::

     128	 1.5138837e+00	 1.3003718e-01	 1.5685962e+00	 1.2380951e-01	[ 1.3357135e+00]	 2.2066903e-01


.. parsed-literal::

     129	 1.5154806e+00	 1.2984052e-01	 1.5701174e+00	 1.2362363e-01	[ 1.3416311e+00]	 2.0767140e-01
     130	 1.5173069e+00	 1.2980583e-01	 1.5719134e+00	 1.2329928e-01	  1.3390913e+00 	 1.9799829e-01


.. parsed-literal::

     131	 1.5184589e+00	 1.2974818e-01	 1.5730998e+00	 1.2307264e-01	  1.3365057e+00 	 2.0766044e-01


.. parsed-literal::

     132	 1.5194252e+00	 1.2969441e-01	 1.5740939e+00	 1.2302466e-01	  1.3351757e+00 	 2.0423198e-01
     133	 1.5199713e+00	 1.2973035e-01	 1.5748272e+00	 1.2265837e-01	  1.3388252e+00 	 1.9482327e-01


.. parsed-literal::

     134	 1.5221036e+00	 1.2958397e-01	 1.5768668e+00	 1.2294775e-01	  1.3392570e+00 	 2.1076250e-01


.. parsed-literal::

     135	 1.5229448e+00	 1.2955833e-01	 1.5776865e+00	 1.2299496e-01	  1.3412154e+00 	 2.1255279e-01


.. parsed-literal::

     136	 1.5241968e+00	 1.2950393e-01	 1.5789502e+00	 1.2298482e-01	[ 1.3433223e+00]	 2.0971251e-01


.. parsed-literal::

     137	 1.5261928e+00	 1.2940791e-01	 1.5809673e+00	 1.2274426e-01	  1.3428806e+00 	 2.0381069e-01


.. parsed-literal::

     138	 1.5273981e+00	 1.2917929e-01	 1.5822850e+00	 1.2281188e-01	  1.3422253e+00 	 3.1791639e-01


.. parsed-literal::

     139	 1.5291742e+00	 1.2907271e-01	 1.5840923e+00	 1.2235648e-01	  1.3394540e+00 	 2.1566176e-01
     140	 1.5303412e+00	 1.2900132e-01	 1.5852993e+00	 1.2203299e-01	  1.3372264e+00 	 1.7558670e-01


.. parsed-literal::

     141	 1.5318978e+00	 1.2889785e-01	 1.5869309e+00	 1.2182385e-01	  1.3350948e+00 	 1.8911958e-01


.. parsed-literal::

     142	 1.5329812e+00	 1.2894998e-01	 1.5881694e+00	 1.2111647e-01	  1.3305377e+00 	 2.1094251e-01
     143	 1.5345098e+00	 1.2890100e-01	 1.5896722e+00	 1.2137535e-01	  1.3328144e+00 	 1.8145418e-01


.. parsed-literal::

     144	 1.5357359e+00	 1.2892224e-01	 1.5908814e+00	 1.2156408e-01	  1.3344275e+00 	 1.8116021e-01
     145	 1.5369967e+00	 1.2890398e-01	 1.5921254e+00	 1.2154105e-01	  1.3335708e+00 	 1.8086624e-01


.. parsed-literal::

     146	 1.5382738e+00	 1.2883783e-01	 1.5934124e+00	 1.2144998e-01	  1.3267646e+00 	 2.0455790e-01
     147	 1.5402791e+00	 1.2867462e-01	 1.5953794e+00	 1.2122542e-01	  1.3262055e+00 	 1.9198942e-01


.. parsed-literal::

     148	 1.5413110e+00	 1.2854618e-01	 1.5963919e+00	 1.2104382e-01	  1.3259047e+00 	 2.0673037e-01


.. parsed-literal::

     149	 1.5426592e+00	 1.2832356e-01	 1.5977610e+00	 1.2092019e-01	  1.3230781e+00 	 2.1339774e-01


.. parsed-literal::

     150	 1.5432640e+00	 1.2810040e-01	 1.5985169e+00	 1.2083823e-01	  1.3220437e+00 	 2.2057271e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.19 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd4f0f20580>



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
    CPU times: user 1.81 s, sys: 39.9 ms, total: 1.85 s
    Wall time: 616 ms


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

