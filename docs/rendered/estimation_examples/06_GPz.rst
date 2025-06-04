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
       1	-3.5033682e-01	 3.2230042e-01	-3.4057316e-01	 3.1501979e-01	[-3.2524068e-01]	 4.7228265e-01


.. parsed-literal::

       2	-2.7642212e-01	 3.1079450e-01	-2.5163039e-01	 3.0214857e-01	[-2.2338183e-01]	 2.2984505e-01


.. parsed-literal::

       3	-2.2777156e-01	 2.8816289e-01	-1.8288441e-01	 2.8273137e-01	[-1.5196077e-01]	 2.7188730e-01


.. parsed-literal::

       4	-1.9180812e-01	 2.7105750e-01	-1.4362478e-01	 2.6722831e-01	[-1.1446393e-01]	 2.9495788e-01


.. parsed-literal::

       5	-1.2718010e-01	 2.6022446e-01	-9.2954069e-02	 2.5634919e-01	[-6.3619545e-02]	 2.1656394e-01


.. parsed-literal::

       6	-8.1605951e-02	 2.5470052e-01	-5.2155147e-02	 2.4828609e-01	[-2.7454283e-02]	 2.1063638e-01
       7	-6.3031026e-02	 2.5135910e-01	-3.8720202e-02	 2.4383834e-01	[-8.4661648e-03]	 1.7936873e-01


.. parsed-literal::

       8	-4.8633945e-02	 2.4888413e-01	-2.8535403e-02	 2.4031925e-01	[ 7.4893530e-03]	 1.8101001e-01


.. parsed-literal::

       9	-3.6493993e-02	 2.4634816e-01	-1.9077274e-02	 2.3662880e-01	[ 1.7173351e-02]	 2.1117520e-01


.. parsed-literal::

      10	-2.6232621e-02	 2.4473916e-01	-1.0942005e-02	 2.3511335e-01	[ 2.9053815e-02]	 2.0435429e-01


.. parsed-literal::

      11	-2.1912723e-02	 2.4395800e-01	-7.0398983e-03	 2.3449967e-01	[ 3.1391781e-02]	 2.1968174e-01
      12	-1.7479305e-02	 2.4315169e-01	-3.1888467e-03	 2.3339122e-01	[ 3.5440847e-02]	 1.9863153e-01


.. parsed-literal::

      13	-1.2438449e-02	 2.4220497e-01	 1.6795456e-03	 2.3262522e-01	[ 4.0030196e-02]	 2.1535969e-01


.. parsed-literal::

      14	 6.7423756e-02	 2.3047052e-01	 8.6743607e-02	 2.2222693e-01	[ 1.2007795e-01]	 3.2304835e-01


.. parsed-literal::

      15	 9.4173368e-02	 2.2980391e-01	 1.1660915e-01	 2.1953861e-01	[ 1.3987325e-01]	 3.2173896e-01


.. parsed-literal::

      16	 1.4931411e-01	 2.2353300e-01	 1.7131546e-01	 2.1177628e-01	[ 1.9715327e-01]	 2.1917176e-01


.. parsed-literal::

      17	 2.5061715e-01	 2.1725376e-01	 2.7792621e-01	 2.0553110e-01	[ 3.1027327e-01]	 2.1237445e-01


.. parsed-literal::

      18	 3.2722600e-01	 2.1129014e-01	 3.5858605e-01	 1.9827679e-01	[ 4.0560130e-01]	 2.0700932e-01


.. parsed-literal::

      19	 3.9289271e-01	 2.0762323e-01	 4.2662265e-01	 2.0015628e-01	[ 4.4979557e-01]	 2.1585059e-01


.. parsed-literal::

      20	 4.4986271e-01	 2.0259670e-01	 4.8450941e-01	 1.9524173e-01	[ 5.0357592e-01]	 2.1326852e-01


.. parsed-literal::

      21	 5.3041186e-01	 1.9898083e-01	 5.6642441e-01	 1.8965623e-01	[ 5.8847292e-01]	 2.1629739e-01


.. parsed-literal::

      22	 6.1146816e-01	 1.9567179e-01	 6.5031993e-01	 1.8512210e-01	[ 6.5151663e-01]	 2.0946932e-01
      23	 6.5485850e-01	 1.9277046e-01	 6.9502426e-01	 1.7855276e-01	[ 6.9033364e-01]	 1.9666672e-01


.. parsed-literal::

      24	 6.8604824e-01	 1.8934843e-01	 7.2546302e-01	 1.7487384e-01	[ 7.2130350e-01]	 2.0666814e-01
      25	 7.2392613e-01	 1.8886695e-01	 7.6277026e-01	 1.7263669e-01	[ 7.5262370e-01]	 2.0020080e-01


.. parsed-literal::

      26	 7.5908831e-01	 1.9174712e-01	 7.9676883e-01	 1.7564611e-01	[ 7.8388224e-01]	 1.8832564e-01


.. parsed-literal::

      27	 7.9216638e-01	 1.9406793e-01	 8.3074922e-01	 1.7778989e-01	[ 8.1341866e-01]	 2.1231651e-01
      28	 8.1009439e-01	 1.9898517e-01	 8.5117955e-01	 1.7953479e-01	[ 8.2702822e-01]	 1.8592882e-01


.. parsed-literal::

      29	 8.3745531e-01	 1.9419205e-01	 8.7804518e-01	 1.7333264e-01	[ 8.6304986e-01]	 1.9171524e-01


.. parsed-literal::

      30	 8.5822272e-01	 1.9064721e-01	 8.9857037e-01	 1.6783316e-01	[ 8.9201330e-01]	 2.2237110e-01


.. parsed-literal::

      31	 8.9002054e-01	 1.8669997e-01	 9.3086789e-01	 1.6294685e-01	[ 9.2876086e-01]	 2.2608161e-01


.. parsed-literal::

      32	 9.0528738e-01	 1.8165121e-01	 9.4692034e-01	 1.5927637e-01	[ 9.4966560e-01]	 2.1776509e-01
      33	 9.2472960e-01	 1.8087334e-01	 9.6627325e-01	 1.5951238e-01	[ 9.7238212e-01]	 2.0910501e-01


.. parsed-literal::

      34	 9.3754090e-01	 1.7888030e-01	 9.7961475e-01	 1.5734062e-01	[ 9.8491293e-01]	 2.0605588e-01
      35	 9.5288235e-01	 1.7602603e-01	 9.9541605e-01	 1.5540711e-01	[ 1.0007807e+00]	 1.8642497e-01


.. parsed-literal::

      36	 9.7313429e-01	 1.7228094e-01	 1.0165089e+00	 1.5437684e-01	[ 1.0248470e+00]	 2.2110462e-01
      37	 9.9089882e-01	 1.6912624e-01	 1.0355079e+00	 1.4889883e-01	[ 1.0442733e+00]	 1.6921258e-01


.. parsed-literal::

      38	 1.0046057e+00	 1.6785867e-01	 1.0493686e+00	 1.4762683e-01	[ 1.0562251e+00]	 1.7563844e-01


.. parsed-literal::

      39	 1.0160586e+00	 1.6690134e-01	 1.0607899e+00	 1.4595658e-01	[ 1.0645142e+00]	 2.1313405e-01


.. parsed-literal::

      40	 1.0323022e+00	 1.6542105e-01	 1.0778207e+00	 1.4208330e-01	[ 1.0763582e+00]	 2.0958304e-01
      41	 1.0450782e+00	 1.6469511e-01	 1.0909946e+00	 1.4025032e-01	[ 1.0880253e+00]	 1.7769766e-01


.. parsed-literal::

      42	 1.0583151e+00	 1.6414292e-01	 1.1042636e+00	 1.3847936e-01	[ 1.1040803e+00]	 2.0079780e-01


.. parsed-literal::

      43	 1.0768487e+00	 1.6290169e-01	 1.1232265e+00	 1.3626867e-01	[ 1.1276571e+00]	 2.0656323e-01
      44	 1.0872001e+00	 1.5969732e-01	 1.1340500e+00	 1.3430160e-01	  1.1220112e+00 	 1.9772935e-01


.. parsed-literal::

      45	 1.0984191e+00	 1.5947560e-01	 1.1451101e+00	 1.3259268e-01	[ 1.1401571e+00]	 2.1049094e-01


.. parsed-literal::

      46	 1.1067018e+00	 1.5841544e-01	 1.1531894e+00	 1.3252245e-01	[ 1.1469780e+00]	 2.1104479e-01


.. parsed-literal::

      47	 1.1185171e+00	 1.5607344e-01	 1.1655805e+00	 1.3202967e-01	[ 1.1526266e+00]	 2.0921111e-01


.. parsed-literal::

      48	 1.1307450e+00	 1.5324025e-01	 1.1785833e+00	 1.3115699e-01	[ 1.1605716e+00]	 2.1012735e-01


.. parsed-literal::

      49	 1.1360197e+00	 1.5136051e-01	 1.1845745e+00	 1.2833081e-01	  1.1594808e+00 	 2.0816684e-01


.. parsed-literal::

      50	 1.1483873e+00	 1.5097977e-01	 1.1964339e+00	 1.2780053e-01	[ 1.1767432e+00]	 2.1140552e-01


.. parsed-literal::

      51	 1.1548429e+00	 1.5046625e-01	 1.2026794e+00	 1.2648982e-01	[ 1.1873612e+00]	 2.1399355e-01
      52	 1.1655665e+00	 1.4929837e-01	 1.2134899e+00	 1.2387860e-01	[ 1.2092031e+00]	 1.8290973e-01


.. parsed-literal::

      53	 1.1736659e+00	 1.4820725e-01	 1.2216768e+00	 1.2168395e-01	[ 1.2215528e+00]	 1.8126464e-01
      54	 1.1795175e+00	 1.4759547e-01	 1.2277007e+00	 1.2101384e-01	[ 1.2259852e+00]	 1.7829728e-01


.. parsed-literal::

      55	 1.1857707e+00	 1.4701011e-01	 1.2342931e+00	 1.2076137e-01	[ 1.2305725e+00]	 1.8362975e-01
      56	 1.1926528e+00	 1.4613092e-01	 1.2416086e+00	 1.2035510e-01	  1.2304573e+00 	 1.9005466e-01


.. parsed-literal::

      57	 1.1999031e+00	 1.4530091e-01	 1.2491327e+00	 1.1950715e-01	[ 1.2386317e+00]	 1.7608309e-01
      58	 1.2063900e+00	 1.4468759e-01	 1.2555395e+00	 1.1857776e-01	[ 1.2466022e+00]	 1.9872284e-01


.. parsed-literal::

      59	 1.2160683e+00	 1.4368124e-01	 1.2651686e+00	 1.1762174e-01	[ 1.2604853e+00]	 2.0499516e-01
      60	 1.2241514e+00	 1.4312528e-01	 1.2733622e+00	 1.1712741e-01	[ 1.2729721e+00]	 1.9593906e-01


.. parsed-literal::

      61	 1.2307895e+00	 1.4252457e-01	 1.2800161e+00	 1.1708438e-01	[ 1.2798700e+00]	 2.0902705e-01
      62	 1.2371056e+00	 1.4192980e-01	 1.2864584e+00	 1.1707768e-01	[ 1.2847662e+00]	 1.9772720e-01


.. parsed-literal::

      63	 1.2430313e+00	 1.4118885e-01	 1.2926410e+00	 1.1672473e-01	[ 1.2887649e+00]	 2.0931220e-01


.. parsed-literal::

      64	 1.2503200e+00	 1.4038788e-01	 1.3002955e+00	 1.1607147e-01	[ 1.2949661e+00]	 2.0878577e-01
      65	 1.2580091e+00	 1.3985474e-01	 1.3080866e+00	 1.1572979e-01	[ 1.3015864e+00]	 1.8785715e-01


.. parsed-literal::

      66	 1.2644794e+00	 1.3957698e-01	 1.3145373e+00	 1.1537647e-01	[ 1.3077853e+00]	 2.0712996e-01


.. parsed-literal::

      67	 1.2734293e+00	 1.3917753e-01	 1.3236339e+00	 1.1517249e-01	[ 1.3155143e+00]	 2.0913172e-01
      68	 1.2785416e+00	 1.3848901e-01	 1.3292521e+00	 1.1478414e-01	[ 1.3249532e+00]	 1.9831967e-01


.. parsed-literal::

      69	 1.2874314e+00	 1.3815850e-01	 1.3379819e+00	 1.1468572e-01	[ 1.3323534e+00]	 2.1119404e-01
      70	 1.2936451e+00	 1.3780755e-01	 1.3442313e+00	 1.1459594e-01	[ 1.3369287e+00]	 1.9312596e-01


.. parsed-literal::

      71	 1.3010245e+00	 1.3729209e-01	 1.3517879e+00	 1.1435383e-01	[ 1.3433334e+00]	 1.7958403e-01
      72	 1.3041880e+00	 1.3677753e-01	 1.3552685e+00	 1.1388777e-01	  1.3399598e+00 	 1.8909264e-01


.. parsed-literal::

      73	 1.3126376e+00	 1.3614087e-01	 1.3636318e+00	 1.1346761e-01	[ 1.3519045e+00]	 1.8207836e-01
      74	 1.3158816e+00	 1.3599344e-01	 1.3667836e+00	 1.1308357e-01	[ 1.3553887e+00]	 2.0165610e-01


.. parsed-literal::

      75	 1.3197174e+00	 1.3563603e-01	 1.3706833e+00	 1.1267202e-01	[ 1.3589532e+00]	 1.9302630e-01
      76	 1.3242503e+00	 1.3545519e-01	 1.3755113e+00	 1.1218296e-01	[ 1.3605334e+00]	 1.9886327e-01


.. parsed-literal::

      77	 1.3292632e+00	 1.3524377e-01	 1.3805417e+00	 1.1218110e-01	[ 1.3637351e+00]	 2.1498942e-01
      78	 1.3336559e+00	 1.3489065e-01	 1.3850871e+00	 1.1239576e-01	[ 1.3671881e+00]	 1.7135644e-01


.. parsed-literal::

      79	 1.3374387e+00	 1.3472999e-01	 1.3889277e+00	 1.1270812e-01	[ 1.3700761e+00]	 1.7261481e-01


.. parsed-literal::

      80	 1.3440362e+00	 1.3442508e-01	 1.3957292e+00	 1.1318309e-01	[ 1.3725645e+00]	 2.0405674e-01


.. parsed-literal::

      81	 1.3504090e+00	 1.3428759e-01	 1.4021856e+00	 1.1402785e-01	[ 1.3730151e+00]	 2.0371270e-01


.. parsed-literal::

      82	 1.3549111e+00	 1.3415033e-01	 1.4066363e+00	 1.1352497e-01	[ 1.3770848e+00]	 2.1544147e-01
      83	 1.3600108e+00	 1.3378451e-01	 1.4118762e+00	 1.1281861e-01	[ 1.3789365e+00]	 1.9855094e-01


.. parsed-literal::

      84	 1.3653497e+00	 1.3355121e-01	 1.4173817e+00	 1.1231117e-01	[ 1.3803533e+00]	 2.0834541e-01


.. parsed-literal::

      85	 1.3723066e+00	 1.3331760e-01	 1.4247300e+00	 1.1186524e-01	  1.3792924e+00 	 2.0214939e-01
      86	 1.3771293e+00	 1.3302413e-01	 1.4294574e+00	 1.1164676e-01	  1.3795242e+00 	 1.8169188e-01


.. parsed-literal::

      87	 1.3804101e+00	 1.3283806e-01	 1.4326320e+00	 1.1166139e-01	[ 1.3835343e+00]	 1.8278265e-01
      88	 1.3839520e+00	 1.3262505e-01	 1.4362414e+00	 1.1175435e-01	[ 1.3852213e+00]	 2.0015168e-01


.. parsed-literal::

      89	 1.3889216e+00	 1.3228175e-01	 1.4413717e+00	 1.1156425e-01	[ 1.3886072e+00]	 2.0837545e-01


.. parsed-literal::

      90	 1.3949507e+00	 1.3194258e-01	 1.4475829e+00	 1.1122704e-01	[ 1.3899245e+00]	 2.0563793e-01
      91	 1.3987800e+00	 1.3156516e-01	 1.4516024e+00	 1.1108283e-01	  1.3898302e+00 	 1.8505383e-01


.. parsed-literal::

      92	 1.4021722e+00	 1.3133344e-01	 1.4548092e+00	 1.1073731e-01	[ 1.3926209e+00]	 2.1863651e-01


.. parsed-literal::

      93	 1.4052929e+00	 1.3098797e-01	 1.4578717e+00	 1.1041935e-01	[ 1.3948141e+00]	 2.1372581e-01


.. parsed-literal::

      94	 1.4095423e+00	 1.3041440e-01	 1.4621493e+00	 1.1012690e-01	[ 1.3967002e+00]	 2.1142054e-01
      95	 1.4108403e+00	 1.2991435e-01	 1.4637721e+00	 1.0972921e-01	[ 1.3972848e+00]	 1.8222857e-01


.. parsed-literal::

      96	 1.4161540e+00	 1.2971231e-01	 1.4688900e+00	 1.0983618e-01	[ 1.3998388e+00]	 1.9952130e-01
      97	 1.4181991e+00	 1.2964301e-01	 1.4709465e+00	 1.0995522e-01	[ 1.4001017e+00]	 1.8131423e-01


.. parsed-literal::

      98	 1.4220339e+00	 1.2951929e-01	 1.4748621e+00	 1.1013195e-01	  1.3996420e+00 	 2.0858049e-01


.. parsed-literal::

      99	 1.4250018e+00	 1.2951898e-01	 1.4779823e+00	 1.1083521e-01	  1.3950414e+00 	 2.0842624e-01


.. parsed-literal::

     100	 1.4287763e+00	 1.2946200e-01	 1.4817224e+00	 1.1078553e-01	  1.3969414e+00 	 2.1830344e-01
     101	 1.4312641e+00	 1.2942442e-01	 1.4841917e+00	 1.1086066e-01	  1.3979234e+00 	 1.9571304e-01


.. parsed-literal::

     102	 1.4337606e+00	 1.2943770e-01	 1.4867309e+00	 1.1113400e-01	  1.3982660e+00 	 2.1546268e-01


.. parsed-literal::

     103	 1.4372937e+00	 1.2946647e-01	 1.4903250e+00	 1.1175051e-01	  1.3980492e+00 	 2.1161151e-01
     104	 1.4404361e+00	 1.2947491e-01	 1.4936239e+00	 1.1237037e-01	[ 1.4016006e+00]	 1.9762349e-01


.. parsed-literal::

     105	 1.4441734e+00	 1.2922868e-01	 1.4973084e+00	 1.1237336e-01	[ 1.4019773e+00]	 2.1018267e-01


.. parsed-literal::

     106	 1.4463540e+00	 1.2900204e-01	 1.4995365e+00	 1.1230358e-01	[ 1.4030760e+00]	 2.0856500e-01
     107	 1.4475164e+00	 1.2873917e-01	 1.5007654e+00	 1.1201678e-01	  1.4028520e+00 	 1.9826770e-01


.. parsed-literal::

     108	 1.4490816e+00	 1.2867165e-01	 1.5023706e+00	 1.1204177e-01	[ 1.4041189e+00]	 1.8976903e-01
     109	 1.4518095e+00	 1.2850493e-01	 1.5052333e+00	 1.1204883e-01	[ 1.4058878e+00]	 2.0162439e-01


.. parsed-literal::

     110	 1.4536777e+00	 1.2838796e-01	 1.5071452e+00	 1.1194826e-01	[ 1.4070040e+00]	 2.0178151e-01
     111	 1.4545031e+00	 1.2810202e-01	 1.5083455e+00	 1.1207048e-01	  1.3993822e+00 	 1.9959831e-01


.. parsed-literal::

     112	 1.4589378e+00	 1.2807564e-01	 1.5125554e+00	 1.1179253e-01	  1.4051076e+00 	 2.0309591e-01
     113	 1.4599170e+00	 1.2810989e-01	 1.5134533e+00	 1.1183450e-01	  1.4057359e+00 	 1.9958782e-01


.. parsed-literal::

     114	 1.4628539e+00	 1.2817008e-01	 1.5165022e+00	 1.1213742e-01	  1.4022479e+00 	 2.1315718e-01
     115	 1.4641718e+00	 1.2831569e-01	 1.5180109e+00	 1.1246174e-01	  1.3953950e+00 	 1.9868469e-01


.. parsed-literal::

     116	 1.4662459e+00	 1.2827189e-01	 1.5200285e+00	 1.1237737e-01	  1.3984177e+00 	 2.1081376e-01


.. parsed-literal::

     117	 1.4676991e+00	 1.2824488e-01	 1.5215076e+00	 1.1234808e-01	  1.3989231e+00 	 2.0276546e-01


.. parsed-literal::

     118	 1.4695965e+00	 1.2821914e-01	 1.5234252e+00	 1.1240949e-01	  1.3976471e+00 	 2.0742440e-01


.. parsed-literal::

     119	 1.4727568e+00	 1.2811414e-01	 1.5265892e+00	 1.1253106e-01	  1.3940274e+00 	 2.0932221e-01


.. parsed-literal::

     120	 1.4745250e+00	 1.2794887e-01	 1.5283914e+00	 1.1270258e-01	  1.3888258e+00 	 3.2308340e-01


.. parsed-literal::

     121	 1.4772752e+00	 1.2783134e-01	 1.5311140e+00	 1.1272461e-01	  1.3860324e+00 	 2.1183896e-01
     122	 1.4797845e+00	 1.2763023e-01	 1.5336209e+00	 1.1264228e-01	  1.3836371e+00 	 1.8043470e-01


.. parsed-literal::

     123	 1.4816352e+00	 1.2749693e-01	 1.5355594e+00	 1.1233278e-01	  1.3815083e+00 	 2.0399809e-01


.. parsed-literal::

     124	 1.4834682e+00	 1.2741351e-01	 1.5373934e+00	 1.1223371e-01	  1.3818164e+00 	 2.0308089e-01


.. parsed-literal::

     125	 1.4847497e+00	 1.2743241e-01	 1.5386816e+00	 1.1216145e-01	  1.3822553e+00 	 2.1669793e-01
     126	 1.4869555e+00	 1.2754823e-01	 1.5409732e+00	 1.1207083e-01	  1.3815871e+00 	 1.9756269e-01


.. parsed-literal::

     127	 1.4885305e+00	 1.2753857e-01	 1.5426112e+00	 1.1200295e-01	  1.3732994e+00 	 2.1096635e-01


.. parsed-literal::

     128	 1.4903978e+00	 1.2753599e-01	 1.5444952e+00	 1.1206920e-01	  1.3698068e+00 	 2.1085715e-01


.. parsed-literal::

     129	 1.4923824e+00	 1.2743858e-01	 1.5464577e+00	 1.1191677e-01	  1.3629423e+00 	 2.1879148e-01


.. parsed-literal::

     130	 1.4937032e+00	 1.2731686e-01	 1.5477598e+00	 1.1193017e-01	  1.3601914e+00 	 2.1210504e-01
     131	 1.4954413e+00	 1.2722608e-01	 1.5495153e+00	 1.1175814e-01	  1.3542364e+00 	 1.9948673e-01


.. parsed-literal::

     132	 1.4967389e+00	 1.2706090e-01	 1.5510396e+00	 1.1165364e-01	  1.3393335e+00 	 2.1074176e-01


.. parsed-literal::

     133	 1.4984200e+00	 1.2707709e-01	 1.5526869e+00	 1.1160900e-01	  1.3391786e+00 	 2.1344233e-01


.. parsed-literal::

     134	 1.4992211e+00	 1.2707681e-01	 1.5535105e+00	 1.1163386e-01	  1.3389830e+00 	 2.1882963e-01


.. parsed-literal::

     135	 1.5005646e+00	 1.2702369e-01	 1.5549089e+00	 1.1170656e-01	  1.3368033e+00 	 2.1979213e-01


.. parsed-literal::

     136	 1.5030415e+00	 1.2692812e-01	 1.5574526e+00	 1.1182376e-01	  1.3281452e+00 	 2.0539856e-01


.. parsed-literal::

     137	 1.5044904e+00	 1.2688891e-01	 1.5589379e+00	 1.1180163e-01	  1.3273349e+00 	 3.0944276e-01


.. parsed-literal::

     138	 1.5065799e+00	 1.2683799e-01	 1.5610153e+00	 1.1190200e-01	  1.3228552e+00 	 2.0171785e-01


.. parsed-literal::

     139	 1.5078799e+00	 1.2682359e-01	 1.5622688e+00	 1.1189418e-01	  1.3207254e+00 	 2.1113443e-01


.. parsed-literal::

     140	 1.5092510e+00	 1.2679694e-01	 1.5636182e+00	 1.1191410e-01	  1.3217895e+00 	 2.1372318e-01


.. parsed-literal::

     141	 1.5108150e+00	 1.2673745e-01	 1.5652305e+00	 1.1171888e-01	  1.3191212e+00 	 2.1362376e-01


.. parsed-literal::

     142	 1.5123239e+00	 1.2664926e-01	 1.5667430e+00	 1.1173011e-01	  1.3131248e+00 	 2.0855737e-01


.. parsed-literal::

     143	 1.5138483e+00	 1.2654174e-01	 1.5683376e+00	 1.1188056e-01	  1.3114166e+00 	 2.0603800e-01
     144	 1.5150614e+00	 1.2645499e-01	 1.5696391e+00	 1.1189331e-01	  1.3031306e+00 	 2.0228887e-01


.. parsed-literal::

     145	 1.5164326e+00	 1.2638002e-01	 1.5710073e+00	 1.1215908e-01	  1.3014564e+00 	 2.0978737e-01


.. parsed-literal::

     146	 1.5177364e+00	 1.2631858e-01	 1.5722802e+00	 1.1244472e-01	  1.2981282e+00 	 2.0559669e-01


.. parsed-literal::

     147	 1.5189420e+00	 1.2625380e-01	 1.5734538e+00	 1.1263985e-01	  1.2938162e+00 	 2.1067309e-01


.. parsed-literal::

     148	 1.5193427e+00	 1.2615530e-01	 1.5739483e+00	 1.1300830e-01	  1.2776403e+00 	 2.0632958e-01


.. parsed-literal::

     149	 1.5215228e+00	 1.2607831e-01	 1.5760290e+00	 1.1300411e-01	  1.2783988e+00 	 2.1715069e-01


.. parsed-literal::

     150	 1.5221507e+00	 1.2603937e-01	 1.5766594e+00	 1.1295410e-01	  1.2776609e+00 	 2.0752621e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.05 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f67c8b13490>



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
    CPU times: user 1.82 s, sys: 56 ms, total: 1.87 s
    Wall time: 589 ms


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

