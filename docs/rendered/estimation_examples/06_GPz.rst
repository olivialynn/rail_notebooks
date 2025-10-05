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
       1	-3.4684494e-01	 3.2172201e-01	-3.3716920e-01	 3.1539622e-01	[-3.2552198e-01]	 4.8478508e-01


.. parsed-literal::

       2	-2.7698389e-01	 3.1135850e-01	-2.5314563e-01	 3.0627144e-01	[-2.3790966e-01]	 2.2867322e-01


.. parsed-literal::

       3	-2.3354439e-01	 2.9053840e-01	-1.9114591e-01	 2.8675127e-01	[-1.7660934e-01]	 2.8632402e-01


.. parsed-literal::

       4	-1.9198294e-01	 2.6692269e-01	-1.4912884e-01	 2.6457724e-01	[-1.3441968e-01]	 2.0428491e-01


.. parsed-literal::

       5	-1.0500589e-01	 2.5722173e-01	-7.2484106e-02	 2.5285383e-01	[-5.2489500e-02]	 2.0482588e-01


.. parsed-literal::

       6	-7.1573841e-02	 2.5223670e-01	-4.2672300e-02	 2.4829026e-01	[-2.8551648e-02]	 2.1539879e-01


.. parsed-literal::

       7	-5.4406554e-02	 2.4949820e-01	-3.0916431e-02	 2.4617534e-01	[-1.7829880e-02]	 2.0093203e-01


.. parsed-literal::

       8	-4.1133333e-02	 2.4724148e-01	-2.1437535e-02	 2.4436418e-01	[-9.8412265e-03]	 2.1616268e-01


.. parsed-literal::

       9	-2.9554038e-02	 2.4508970e-01	-1.2239221e-02	 2.4220141e-01	[-6.2242998e-04]	 2.1811509e-01


.. parsed-literal::

      10	-1.8837221e-02	 2.4309956e-01	-3.3574517e-03	 2.4033982e-01	[ 7.6676137e-03]	 2.1499562e-01


.. parsed-literal::

      11	-1.4801985e-02	 2.4262633e-01	-6.1859045e-04	 2.3949128e-01	[ 1.1867406e-02]	 2.1223044e-01


.. parsed-literal::

      12	-1.1743407e-02	 2.4191694e-01	 2.2737781e-03	 2.3888586e-01	[ 1.4406925e-02]	 2.1019959e-01
      13	-8.1891990e-03	 2.4120895e-01	 5.8151868e-03	 2.3796144e-01	[ 1.9008088e-02]	 1.7852092e-01


.. parsed-literal::

      14	-2.8664856e-03	 2.4004806e-01	 1.1686497e-02	 2.3646008e-01	[ 2.5632293e-02]	 1.8268108e-01
      15	 4.6236232e-02	 2.3082638e-01	 6.3356731e-02	 2.2566623e-01	[ 7.9982134e-02]	 1.8172479e-01


.. parsed-literal::

      16	 6.6896336e-02	 2.2699725e-01	 8.5214804e-02	 2.2293636e-01	[ 9.5464056e-02]	 3.0033660e-01


.. parsed-literal::

      17	 1.2155024e-01	 2.2229451e-01	 1.4201746e-01	 2.1822002e-01	[ 1.5369433e-01]	 2.0414472e-01
      18	 2.4105734e-01	 2.1960088e-01	 2.7179803e-01	 2.1437147e-01	[ 2.9753346e-01]	 1.8424249e-01


.. parsed-literal::

      19	 2.8413843e-01	 2.1735997e-01	 3.1487088e-01	 2.1352955e-01	[ 3.3601201e-01]	 2.0621300e-01


.. parsed-literal::

      20	 3.2515433e-01	 2.1464058e-01	 3.5491914e-01	 2.1173757e-01	[ 3.7157191e-01]	 2.1329308e-01


.. parsed-literal::

      21	 3.6641009e-01	 2.1003658e-01	 3.9698372e-01	 2.0749943e-01	[ 4.1094295e-01]	 2.1052551e-01
      22	 4.2526514e-01	 2.0630763e-01	 4.5709522e-01	 2.0291321e-01	[ 4.7153908e-01]	 1.9105005e-01


.. parsed-literal::

      23	 5.4330790e-01	 2.0566863e-01	 5.7946192e-01	 1.9952138e-01	[ 6.0504066e-01]	 2.0334482e-01


.. parsed-literal::

      24	 6.0266507e-01	 2.0757350e-01	 6.4230181e-01	 2.0242367e-01	[ 6.2684335e-01]	 2.1846056e-01


.. parsed-literal::

      25	 6.3909527e-01	 2.0308931e-01	 6.7658997e-01	 1.9807416e-01	[ 6.6762166e-01]	 2.1083593e-01


.. parsed-literal::

      26	 6.7557375e-01	 1.9852496e-01	 7.1348770e-01	 1.9397633e-01	[ 7.0409407e-01]	 2.1383405e-01


.. parsed-literal::

      27	 7.0876872e-01	 1.9439910e-01	 7.4640845e-01	 1.9028297e-01	[ 7.4478600e-01]	 3.1278205e-01


.. parsed-literal::

      28	 7.2608845e-01	 1.9568527e-01	 7.6379485e-01	 1.9067865e-01	[ 7.7309826e-01]	 2.0963788e-01


.. parsed-literal::

      29	 7.6731180e-01	 1.9215379e-01	 8.0444537e-01	 1.8958240e-01	[ 8.0118740e-01]	 2.0982003e-01


.. parsed-literal::

      30	 7.8882472e-01	 1.9455107e-01	 8.2621533e-01	 1.9373010e-01	[ 8.2683213e-01]	 2.1330762e-01


.. parsed-literal::

      31	 8.0618468e-01	 1.9718440e-01	 8.4462594e-01	 1.9743513e-01	[ 8.4130119e-01]	 2.0580959e-01


.. parsed-literal::

      32	 8.2539318e-01	 1.9919504e-01	 8.6422947e-01	 2.0007761e-01	[ 8.6346751e-01]	 2.0587873e-01


.. parsed-literal::

      33	 8.4403901e-01	 1.9782721e-01	 8.8389780e-01	 1.9829404e-01	[ 8.8162450e-01]	 2.1419287e-01


.. parsed-literal::

      34	 8.7453554e-01	 1.9489808e-01	 9.1590880e-01	 1.9540449e-01	[ 9.1042670e-01]	 2.0998740e-01


.. parsed-literal::

      35	 8.9499771e-01	 1.9055783e-01	 9.3841033e-01	 1.8992712e-01	[ 9.1924392e-01]	 2.1028852e-01


.. parsed-literal::

      36	 9.1917119e-01	 1.8796199e-01	 9.6169036e-01	 1.8818362e-01	[ 9.4867437e-01]	 2.0694232e-01


.. parsed-literal::

      37	 9.3499138e-01	 1.8467484e-01	 9.7648989e-01	 1.8466507e-01	[ 9.6481762e-01]	 2.0985937e-01
      38	 9.4514853e-01	 1.8254118e-01	 9.8672416e-01	 1.8248043e-01	[ 9.7215238e-01]	 1.9840193e-01


.. parsed-literal::

      39	 9.6436422e-01	 1.7961462e-01	 1.0063270e+00	 1.7929984e-01	[ 9.9240893e-01]	 2.1404791e-01
      40	 9.7381128e-01	 1.8215468e-01	 1.0171631e+00	 1.8002125e-01	[ 1.0069752e+00]	 2.0014286e-01


.. parsed-literal::

      41	 9.8714485e-01	 1.8052113e-01	 1.0304800e+00	 1.7901935e-01	[ 1.0210418e+00]	 2.0885038e-01
      42	 9.9358282e-01	 1.7969259e-01	 1.0370688e+00	 1.7885868e-01	[ 1.0272037e+00]	 1.7537403e-01


.. parsed-literal::

      43	 1.0028612e+00	 1.7959394e-01	 1.0469190e+00	 1.7919204e-01	[ 1.0351351e+00]	 2.0785069e-01


.. parsed-literal::

      44	 1.0139674e+00	 1.7854069e-01	 1.0586030e+00	 1.7920450e-01	[ 1.0394474e+00]	 2.1261597e-01
      45	 1.0217557e+00	 1.7858420e-01	 1.0667673e+00	 1.7936307e-01	[ 1.0485425e+00]	 1.7600608e-01


.. parsed-literal::

      46	 1.0331759e+00	 1.7774113e-01	 1.0784163e+00	 1.7846547e-01	[ 1.0602905e+00]	 1.7394614e-01


.. parsed-literal::

      47	 1.0502749e+00	 1.7626478e-01	 1.0966929e+00	 1.7649542e-01	[ 1.0735550e+00]	 2.1086502e-01
      48	 1.0540224e+00	 1.7321482e-01	 1.1007966e+00	 1.7260932e-01	  1.0671556e+00 	 1.9924259e-01


.. parsed-literal::

      49	 1.0687876e+00	 1.7202135e-01	 1.1144398e+00	 1.7184200e-01	[ 1.0851791e+00]	 1.8473315e-01
      50	 1.0747099e+00	 1.7169239e-01	 1.1204318e+00	 1.7144413e-01	[ 1.0892031e+00]	 1.8424058e-01


.. parsed-literal::

      51	 1.0853055e+00	 1.7069616e-01	 1.1314409e+00	 1.7019094e-01	[ 1.0914338e+00]	 2.0315385e-01
      52	 1.0965364e+00	 1.6958000e-01	 1.1429176e+00	 1.6893189e-01	[ 1.0949198e+00]	 1.8330717e-01


.. parsed-literal::

      53	 1.1040692e+00	 1.6712581e-01	 1.1511069e+00	 1.6768368e-01	  1.0903422e+00 	 2.0498776e-01


.. parsed-literal::

      54	 1.1121933e+00	 1.6695739e-01	 1.1591085e+00	 1.6772103e-01	[ 1.0976177e+00]	 2.1308160e-01


.. parsed-literal::

      55	 1.1197595e+00	 1.6665422e-01	 1.1668473e+00	 1.6760381e-01	[ 1.1034325e+00]	 2.0827031e-01
      56	 1.1279293e+00	 1.6567805e-01	 1.1751631e+00	 1.6715930e-01	[ 1.1087839e+00]	 1.9968653e-01


.. parsed-literal::

      57	 1.1339768e+00	 1.6427181e-01	 1.1814885e+00	 1.6616757e-01	[ 1.1136446e+00]	 3.2410336e-01
      58	 1.1414239e+00	 1.6268268e-01	 1.1889736e+00	 1.6489215e-01	[ 1.1211803e+00]	 1.9770360e-01


.. parsed-literal::

      59	 1.1471740e+00	 1.6144600e-01	 1.1948334e+00	 1.6384240e-01	[ 1.1266285e+00]	 1.8131304e-01
      60	 1.1551452e+00	 1.6006663e-01	 1.2029901e+00	 1.6227066e-01	[ 1.1384143e+00]	 1.9944334e-01


.. parsed-literal::

      61	 1.1604178e+00	 1.5800173e-01	 1.2086622e+00	 1.6096034e-01	[ 1.1450268e+00]	 2.0910621e-01
      62	 1.1679839e+00	 1.5829621e-01	 1.2160619e+00	 1.6070530e-01	[ 1.1548929e+00]	 1.8185544e-01


.. parsed-literal::

      63	 1.1733010e+00	 1.5780862e-01	 1.2215004e+00	 1.6011809e-01	[ 1.1599509e+00]	 1.7758393e-01


.. parsed-literal::

      64	 1.1792302e+00	 1.5788295e-01	 1.2276381e+00	 1.5994663e-01	[ 1.1657188e+00]	 2.1436429e-01
      65	 1.1875528e+00	 1.5705996e-01	 1.2363267e+00	 1.5906560e-01	[ 1.1725569e+00]	 1.7093587e-01


.. parsed-literal::

      66	 1.1951981e+00	 1.5616481e-01	 1.2446973e+00	 1.5792637e-01	[ 1.1771052e+00]	 2.0906186e-01


.. parsed-literal::

      67	 1.2030036e+00	 1.5599170e-01	 1.2526016e+00	 1.5765280e-01	[ 1.1839159e+00]	 2.0848942e-01


.. parsed-literal::

      68	 1.2118343e+00	 1.5479905e-01	 1.2617717e+00	 1.5663769e-01	[ 1.1893959e+00]	 2.2267175e-01


.. parsed-literal::

      69	 1.2193172e+00	 1.5364225e-01	 1.2696673e+00	 1.5555772e-01	[ 1.1911903e+00]	 2.1340513e-01
      70	 1.2260594e+00	 1.5285628e-01	 1.2765800e+00	 1.5483111e-01	[ 1.1929900e+00]	 1.9882631e-01


.. parsed-literal::

      71	 1.2342358e+00	 1.5168383e-01	 1.2849460e+00	 1.5368600e-01	[ 1.1945696e+00]	 2.1565533e-01


.. parsed-literal::

      72	 1.2408396e+00	 1.5100860e-01	 1.2917167e+00	 1.5317116e-01	[ 1.1954159e+00]	 2.1104264e-01
      73	 1.2479830e+00	 1.5029665e-01	 1.2989924e+00	 1.5246213e-01	[ 1.2022578e+00]	 1.8491411e-01


.. parsed-literal::

      74	 1.2534098e+00	 1.4988537e-01	 1.3044236e+00	 1.5231538e-01	[ 1.2100195e+00]	 1.8651152e-01


.. parsed-literal::

      75	 1.2582612e+00	 1.4964606e-01	 1.3093756e+00	 1.5212450e-01	[ 1.2143759e+00]	 2.1657491e-01
      76	 1.2629228e+00	 1.4927085e-01	 1.3139662e+00	 1.5188745e-01	[ 1.2201951e+00]	 2.0028782e-01


.. parsed-literal::

      77	 1.2683862e+00	 1.4846946e-01	 1.3195003e+00	 1.5121610e-01	[ 1.2217342e+00]	 2.1394920e-01


.. parsed-literal::

      78	 1.2742967e+00	 1.4765369e-01	 1.3256402e+00	 1.5047792e-01	  1.2166058e+00 	 2.1458244e-01


.. parsed-literal::

      79	 1.2786597e+00	 1.4691402e-01	 1.3301468e+00	 1.4978147e-01	  1.2171869e+00 	 2.0995355e-01


.. parsed-literal::

      80	 1.2827711e+00	 1.4652368e-01	 1.3343231e+00	 1.4942901e-01	  1.2187588e+00 	 2.1705151e-01


.. parsed-literal::

      81	 1.2877621e+00	 1.4609603e-01	 1.3395382e+00	 1.4895157e-01	[ 1.2217650e+00]	 2.1479845e-01
      82	 1.2916723e+00	 1.4581365e-01	 1.3436061e+00	 1.4839918e-01	[ 1.2262579e+00]	 1.8368268e-01


.. parsed-literal::

      83	 1.2962851e+00	 1.4537965e-01	 1.3482365e+00	 1.4808078e-01	[ 1.2299645e+00]	 1.7729974e-01
      84	 1.3022076e+00	 1.4455715e-01	 1.3542478e+00	 1.4740422e-01	[ 1.2347253e+00]	 1.9221020e-01


.. parsed-literal::

      85	 1.3052776e+00	 1.4385876e-01	 1.3574171e+00	 1.4685180e-01	[ 1.2368775e+00]	 1.6754055e-01
      86	 1.3091314e+00	 1.4336262e-01	 1.3612977e+00	 1.4636591e-01	[ 1.2402205e+00]	 1.7431569e-01


.. parsed-literal::

      87	 1.3140070e+00	 1.4270952e-01	 1.3662537e+00	 1.4573896e-01	[ 1.2406764e+00]	 2.1101689e-01


.. parsed-literal::

      88	 1.3159741e+00	 1.4209579e-01	 1.3686673e+00	 1.4559547e-01	  1.2315401e+00 	 2.0807910e-01


.. parsed-literal::

      89	 1.3219255e+00	 1.4193651e-01	 1.3744664e+00	 1.4527203e-01	  1.2354257e+00 	 2.1010232e-01
      90	 1.3244403e+00	 1.4183725e-01	 1.3770186e+00	 1.4515052e-01	  1.2350247e+00 	 2.0509648e-01


.. parsed-literal::

      91	 1.3288796e+00	 1.4160006e-01	 1.3816237e+00	 1.4491845e-01	  1.2329173e+00 	 2.0331883e-01


.. parsed-literal::

      92	 1.3343223e+00	 1.4134162e-01	 1.3871787e+00	 1.4456995e-01	  1.2328921e+00 	 2.1366930e-01


.. parsed-literal::

      93	 1.3375550e+00	 1.4099856e-01	 1.3906318e+00	 1.4428592e-01	  1.2343897e+00 	 3.3382511e-01
      94	 1.3425316e+00	 1.4084903e-01	 1.3956360e+00	 1.4402129e-01	  1.2371500e+00 	 1.8335366e-01


.. parsed-literal::

      95	 1.3459502e+00	 1.4071807e-01	 1.3990030e+00	 1.4386224e-01	  1.2402352e+00 	 2.1720171e-01


.. parsed-literal::

      96	 1.3502841e+00	 1.4043893e-01	 1.4033976e+00	 1.4369985e-01	[ 1.2418283e+00]	 2.1736145e-01
      97	 1.3537815e+00	 1.4035802e-01	 1.4069626e+00	 1.4361493e-01	  1.2383007e+00 	 1.7845058e-01


.. parsed-literal::

      98	 1.3570126e+00	 1.4002911e-01	 1.4101626e+00	 1.4343567e-01	  1.2411997e+00 	 1.9942069e-01


.. parsed-literal::

      99	 1.3618252e+00	 1.3957337e-01	 1.4150533e+00	 1.4315648e-01	[ 1.2435338e+00]	 2.1186471e-01
     100	 1.3648066e+00	 1.3929475e-01	 1.4180379e+00	 1.4283163e-01	[ 1.2476681e+00]	 1.9775510e-01


.. parsed-literal::

     101	 1.3681527e+00	 1.3927369e-01	 1.4213427e+00	 1.4261668e-01	[ 1.2508279e+00]	 2.1109819e-01
     102	 1.3731626e+00	 1.3927313e-01	 1.4263753e+00	 1.4223207e-01	[ 1.2532265e+00]	 1.8364549e-01


.. parsed-literal::

     103	 1.3759584e+00	 1.3939830e-01	 1.4292218e+00	 1.4197385e-01	[ 1.2567435e+00]	 2.0590973e-01


.. parsed-literal::

     104	 1.3791875e+00	 1.3904502e-01	 1.4325370e+00	 1.4145196e-01	[ 1.2593755e+00]	 2.0103645e-01


.. parsed-literal::

     105	 1.3814581e+00	 1.3893370e-01	 1.4348960e+00	 1.4142449e-01	  1.2592309e+00 	 2.1266699e-01


.. parsed-literal::

     106	 1.3834481e+00	 1.3859640e-01	 1.4368783e+00	 1.4122943e-01	[ 1.2603504e+00]	 2.1105695e-01


.. parsed-literal::

     107	 1.3858351e+00	 1.3823698e-01	 1.4393116e+00	 1.4111301e-01	  1.2598426e+00 	 2.1119618e-01
     108	 1.3886459e+00	 1.3785144e-01	 1.4422111e+00	 1.4095309e-01	  1.2602503e+00 	 2.0121384e-01


.. parsed-literal::

     109	 1.3913741e+00	 1.3765782e-01	 1.4450624e+00	 1.4100018e-01	  1.2595587e+00 	 2.0887399e-01
     110	 1.3935290e+00	 1.3763703e-01	 1.4471854e+00	 1.4090149e-01	[ 1.2613926e+00]	 2.0367503e-01


.. parsed-literal::

     111	 1.3956232e+00	 1.3754678e-01	 1.4492702e+00	 1.4081040e-01	[ 1.2640615e+00]	 2.0713592e-01


.. parsed-literal::

     112	 1.3984645e+00	 1.3732212e-01	 1.4521621e+00	 1.4068794e-01	[ 1.2679510e+00]	 2.0689201e-01


.. parsed-literal::

     113	 1.4011525e+00	 1.3679750e-01	 1.4550811e+00	 1.4046181e-01	[ 1.2784886e+00]	 2.1138382e-01
     114	 1.4049229e+00	 1.3658884e-01	 1.4588554e+00	 1.4032313e-01	[ 1.2824481e+00]	 1.7805505e-01


.. parsed-literal::

     115	 1.4065363e+00	 1.3651476e-01	 1.4604811e+00	 1.4020501e-01	[ 1.2836817e+00]	 1.8008757e-01


.. parsed-literal::

     116	 1.4097863e+00	 1.3632689e-01	 1.4638413e+00	 1.3992510e-01	[ 1.2872515e+00]	 2.0946980e-01
     117	 1.4123319e+00	 1.3649209e-01	 1.4666451e+00	 1.3976988e-01	[ 1.2910999e+00]	 1.7187428e-01


.. parsed-literal::

     118	 1.4154567e+00	 1.3629352e-01	 1.4696922e+00	 1.3953091e-01	[ 1.2953459e+00]	 2.0481157e-01


.. parsed-literal::

     119	 1.4178884e+00	 1.3626957e-01	 1.4720797e+00	 1.3944371e-01	[ 1.2982379e+00]	 2.1166205e-01


.. parsed-literal::

     120	 1.4204856e+00	 1.3620784e-01	 1.4746777e+00	 1.3927744e-01	[ 1.3001727e+00]	 2.1608949e-01


.. parsed-literal::

     121	 1.4227296e+00	 1.3626811e-01	 1.4770450e+00	 1.3899723e-01	  1.2999957e+00 	 2.1530271e-01


.. parsed-literal::

     122	 1.4255107e+00	 1.3609801e-01	 1.4797576e+00	 1.3884005e-01	[ 1.3010519e+00]	 2.1225548e-01


.. parsed-literal::

     123	 1.4275733e+00	 1.3590150e-01	 1.4818418e+00	 1.3866109e-01	[ 1.3013666e+00]	 2.0357108e-01


.. parsed-literal::

     124	 1.4297628e+00	 1.3570506e-01	 1.4840742e+00	 1.3845260e-01	  1.3012717e+00 	 2.1781397e-01
     125	 1.4322863e+00	 1.3554184e-01	 1.4867307e+00	 1.3824733e-01	[ 1.3053782e+00]	 1.9609427e-01


.. parsed-literal::

     126	 1.4355600e+00	 1.3532315e-01	 1.4899608e+00	 1.3788646e-01	[ 1.3066921e+00]	 1.7590380e-01
     127	 1.4375026e+00	 1.3520592e-01	 1.4918779e+00	 1.3764823e-01	[ 1.3089220e+00]	 1.9904518e-01


.. parsed-literal::

     128	 1.4397053e+00	 1.3493176e-01	 1.4940693e+00	 1.3722845e-01	[ 1.3117868e+00]	 2.1014094e-01


.. parsed-literal::

     129	 1.4417306e+00	 1.3445693e-01	 1.4962023e+00	 1.3647761e-01	[ 1.3126300e+00]	 2.0188308e-01


.. parsed-literal::

     130	 1.4451111e+00	 1.3411525e-01	 1.4995105e+00	 1.3606078e-01	[ 1.3150631e+00]	 2.0459485e-01


.. parsed-literal::

     131	 1.4472336e+00	 1.3377035e-01	 1.5016237e+00	 1.3575520e-01	  1.3146780e+00 	 2.1038842e-01


.. parsed-literal::

     132	 1.4491029e+00	 1.3348002e-01	 1.5035742e+00	 1.3551289e-01	  1.3124645e+00 	 2.0101881e-01


.. parsed-literal::

     133	 1.4502677e+00	 1.3288708e-01	 1.5049819e+00	 1.3505095e-01	  1.3090691e+00 	 2.0619655e-01
     134	 1.4525330e+00	 1.3293606e-01	 1.5072370e+00	 1.3513241e-01	  1.3099339e+00 	 1.7412353e-01


.. parsed-literal::

     135	 1.4542682e+00	 1.3293708e-01	 1.5090425e+00	 1.3516595e-01	  1.3101677e+00 	 1.9971776e-01


.. parsed-literal::

     136	 1.4560306e+00	 1.3290446e-01	 1.5109061e+00	 1.3519252e-01	  1.3096656e+00 	 2.1691585e-01


.. parsed-literal::

     137	 1.4589606e+00	 1.3290523e-01	 1.5139959e+00	 1.3529666e-01	  1.3085530e+00 	 2.0833707e-01


.. parsed-literal::

     138	 1.4610493e+00	 1.3318534e-01	 1.5162747e+00	 1.3562018e-01	  1.3023065e+00 	 2.0859480e-01


.. parsed-literal::

     139	 1.4629648e+00	 1.3304522e-01	 1.5180407e+00	 1.3544777e-01	  1.3069501e+00 	 2.0510054e-01
     140	 1.4644119e+00	 1.3302062e-01	 1.5194383e+00	 1.3539553e-01	  1.3085899e+00 	 1.8336344e-01


.. parsed-literal::

     141	 1.4664813e+00	 1.3312882e-01	 1.5215327e+00	 1.3546845e-01	  1.3096253e+00 	 2.0563483e-01


.. parsed-literal::

     142	 1.4682744e+00	 1.3328401e-01	 1.5235915e+00	 1.3547883e-01	  1.3101388e+00 	 2.0624375e-01


.. parsed-literal::

     143	 1.4712000e+00	 1.3358552e-01	 1.5265279e+00	 1.3574954e-01	  1.3117162e+00 	 2.1011376e-01
     144	 1.4726589e+00	 1.3365602e-01	 1.5280340e+00	 1.3579399e-01	  1.3132896e+00 	 1.7994642e-01


.. parsed-literal::

     145	 1.4745149e+00	 1.3373540e-01	 1.5299956e+00	 1.3581212e-01	[ 1.3162287e+00]	 2.0633769e-01


.. parsed-literal::

     146	 1.4765816e+00	 1.3372896e-01	 1.5321910e+00	 1.3564718e-01	[ 1.3209914e+00]	 2.1277070e-01


.. parsed-literal::

     147	 1.4790243e+00	 1.3372345e-01	 1.5346236e+00	 1.3555521e-01	[ 1.3271879e+00]	 2.0859790e-01
     148	 1.4808201e+00	 1.3360476e-01	 1.5363593e+00	 1.3543404e-01	[ 1.3308253e+00]	 2.0776010e-01


.. parsed-literal::

     149	 1.4826950e+00	 1.3335056e-01	 1.5382086e+00	 1.3519204e-01	[ 1.3323545e+00]	 2.1460629e-01


.. parsed-literal::

     150	 1.4842190e+00	 1.3302222e-01	 1.5397899e+00	 1.3508413e-01	  1.3295097e+00 	 2.0501971e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.19 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb414b24a90>



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
    CPU times: user 1.77 s, sys: 50 ms, total: 1.82 s
    Wall time: 576 ms


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

