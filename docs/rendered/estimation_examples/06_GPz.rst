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
       1	-3.5215900e-01	 3.2284440e-01	-3.4250350e-01	 3.1145985e-01	[-3.2212342e-01]	 4.6703410e-01


.. parsed-literal::

       2	-2.7797682e-01	 3.1135230e-01	-2.5363389e-01	 3.0177208e-01	[-2.2561123e-01]	 2.3233461e-01


.. parsed-literal::

       3	-2.3428926e-01	 2.9163593e-01	-1.9345950e-01	 2.8311244e-01	[-1.6053641e-01]	 2.8393865e-01


.. parsed-literal::

       4	-2.0431068e-01	 2.6779361e-01	-1.6518824e-01	 2.6248602e-01	[-1.2545702e-01]	 2.0782113e-01


.. parsed-literal::

       5	-1.1694788e-01	 2.6023303e-01	-8.2305816e-02	 2.4922976e-01	[-3.4012977e-02]	 2.0240927e-01


.. parsed-literal::

       6	-8.1966122e-02	 2.5402124e-01	-4.9927044e-02	 2.4568007e-01	[-1.5561872e-02]	 2.1019268e-01


.. parsed-literal::

       7	-6.4985843e-02	 2.5167317e-01	-3.9933153e-02	 2.4108131e-01	[ 2.8749193e-03]	 2.2482324e-01


.. parsed-literal::

       8	-4.9632067e-02	 2.4902604e-01	-2.8997396e-02	 2.3701395e-01	[ 1.9067364e-02]	 2.1385241e-01
       9	-3.3143131e-02	 2.4593431e-01	-1.5939013e-02	 2.3358843e-01	[ 3.5139770e-02]	 1.8820429e-01


.. parsed-literal::

      10	-2.6319047e-02	 2.4507171e-01	-1.1337854e-02	 2.3483900e-01	  2.9106652e-02 	 2.1408772e-01


.. parsed-literal::

      11	-1.9944253e-02	 2.4360970e-01	-5.3769152e-03	 2.3420430e-01	  3.3429945e-02 	 2.1865559e-01
      12	-1.7166386e-02	 2.4307458e-01	-2.8660165e-03	 2.3370015e-01	[ 3.5879191e-02]	 1.8721485e-01


.. parsed-literal::

      13	-1.3241166e-02	 2.4229962e-01	 1.0307791e-03	 2.3255192e-01	[ 4.0908230e-02]	 1.8953466e-01
      14	-3.1637985e-04	 2.3973864e-01	 1.5152090e-02	 2.2959834e-01	[ 5.5813951e-02]	 1.9023800e-01


.. parsed-literal::

      15	 5.5268517e-02	 2.2572356e-01	 7.5357311e-02	 2.1417953e-01	[ 9.9055996e-02]	 1.8543673e-01


.. parsed-literal::

      16	 2.2156499e-01	 2.2576983e-01	 2.5130305e-01	 2.1865717e-01	[ 2.9130255e-01]	 2.0659280e-01
      17	 2.7325396e-01	 2.1953122e-01	 3.0495361e-01	 2.1216166e-01	[ 3.4482638e-01]	 1.9931698e-01


.. parsed-literal::

      18	 3.3557348e-01	 2.1345292e-01	 3.6757244e-01	 2.0539722e-01	[ 4.0006312e-01]	 2.0308447e-01
      19	 3.7900831e-01	 2.0983900e-01	 4.1145371e-01	 2.0260557e-01	[ 4.4451089e-01]	 1.9963837e-01


.. parsed-literal::

      20	 4.6743152e-01	 2.0552567e-01	 5.0239528e-01	 1.9978925e-01	[ 5.3457578e-01]	 2.1245885e-01


.. parsed-literal::

      21	 5.5113554e-01	 2.1277414e-01	 5.9043498e-01	 2.0479325e-01	[ 6.0908756e-01]	 2.1623850e-01


.. parsed-literal::

      22	 5.6050024e-01	 2.1192584e-01	 5.9746573e-01	 2.0763473e-01	  6.0862366e-01 	 2.0519328e-01
      23	 6.1832945e-01	 2.0545033e-01	 6.5808732e-01	 2.0036829e-01	[ 6.6463423e-01]	 1.9046450e-01


.. parsed-literal::

      24	 6.5615474e-01	 2.0313318e-01	 6.9464832e-01	 1.9663587e-01	[ 7.0953160e-01]	 2.1252203e-01
      25	 6.8498316e-01	 2.0198604e-01	 7.2223305e-01	 1.9369795e-01	[ 7.4386535e-01]	 1.7846894e-01


.. parsed-literal::

      26	 7.2165645e-01	 1.9927624e-01	 7.5935975e-01	 1.9123927e-01	[ 7.7582143e-01]	 2.0001483e-01


.. parsed-literal::

      27	 7.6708363e-01	 2.0169004e-01	 8.0616030e-01	 1.9275245e-01	[ 8.1903981e-01]	 2.0486093e-01
      28	 8.0081914e-01	 2.0758070e-01	 8.4147576e-01	 1.9779420e-01	[ 8.3251379e-01]	 1.6244674e-01


.. parsed-literal::

      29	 8.4411252e-01	 2.0884880e-01	 8.8546976e-01	 1.9684636e-01	[ 8.8445036e-01]	 2.0979548e-01
      30	 8.6741450e-01	 2.0849719e-01	 9.1061488e-01	 1.9675481e-01	[ 8.9553595e-01]	 1.7746782e-01


.. parsed-literal::

      31	 8.8851109e-01	 2.0918800e-01	 9.3124707e-01	 1.9683680e-01	[ 9.2260658e-01]	 2.0355344e-01
      32	 9.1930547e-01	 2.0584491e-01	 9.6213187e-01	 1.9306986e-01	[ 9.6265372e-01]	 1.9975281e-01


.. parsed-literal::

      33	 9.4851318e-01	 2.0051905e-01	 9.9291018e-01	 1.8607206e-01	[ 9.9678611e-01]	 2.0313215e-01
      34	 9.7416171e-01	 1.9631311e-01	 1.0196490e+00	 1.8232281e-01	[ 1.0213639e+00]	 2.0232058e-01


.. parsed-literal::

      35	 9.8694939e-01	 1.9407698e-01	 1.0326367e+00	 1.8011645e-01	[ 1.0345304e+00]	 2.1055079e-01


.. parsed-literal::

      36	 1.0064549e+00	 1.9028109e-01	 1.0530987e+00	 1.7617594e-01	[ 1.0505063e+00]	 2.0391989e-01


.. parsed-literal::

      37	 1.0199482e+00	 1.8525074e-01	 1.0674506e+00	 1.6946633e-01	[ 1.0621490e+00]	 2.0545459e-01
      38	 1.0362971e+00	 1.8347337e-01	 1.0836826e+00	 1.6747765e-01	[ 1.0783480e+00]	 1.7717266e-01


.. parsed-literal::

      39	 1.0490586e+00	 1.8161162e-01	 1.0961622e+00	 1.6556806e-01	[ 1.0890300e+00]	 1.9271159e-01
      40	 1.0673774e+00	 1.7901845e-01	 1.1145643e+00	 1.6330964e-01	[ 1.1018019e+00]	 1.7909312e-01


.. parsed-literal::

      41	 1.0850011e+00	 1.7637439e-01	 1.1321546e+00	 1.6267036e-01	[ 1.1219213e+00]	 1.9775248e-01
      42	 1.1007860e+00	 1.7565422e-01	 1.1484059e+00	 1.6244916e-01	[ 1.1326903e+00]	 1.8949819e-01


.. parsed-literal::

      43	 1.1135110e+00	 1.7591234e-01	 1.1616141e+00	 1.6261959e-01	[ 1.1439175e+00]	 2.0998240e-01
      44	 1.1281383e+00	 1.7508658e-01	 1.1768527e+00	 1.6166102e-01	[ 1.1582929e+00]	 1.7564702e-01


.. parsed-literal::

      45	 1.1390425e+00	 1.7419077e-01	 1.1876747e+00	 1.6080731e-01	[ 1.1613721e+00]	 1.9977307e-01


.. parsed-literal::

      46	 1.1504884e+00	 1.7237598e-01	 1.1990152e+00	 1.5880746e-01	[ 1.1670632e+00]	 2.0129395e-01


.. parsed-literal::

      47	 1.1701332e+00	 1.6918452e-01	 1.2189266e+00	 1.5506018e-01	[ 1.1717803e+00]	 2.1279836e-01
      48	 1.1867514e+00	 1.6527494e-01	 1.2350982e+00	 1.5058303e-01	[ 1.1866113e+00]	 2.0177460e-01


.. parsed-literal::

      49	 1.2064342e+00	 1.6084996e-01	 1.2558160e+00	 1.4639812e-01	[ 1.1927904e+00]	 2.0512176e-01


.. parsed-literal::

      50	 1.2237458e+00	 1.5914166e-01	 1.2733883e+00	 1.4459824e-01	[ 1.2096059e+00]	 2.0251751e-01


.. parsed-literal::

      51	 1.2402052e+00	 1.5512651e-01	 1.2902422e+00	 1.4181959e-01	[ 1.2243640e+00]	 2.1274090e-01


.. parsed-literal::

      52	 1.2565810e+00	 1.5427912e-01	 1.3066010e+00	 1.4116586e-01	[ 1.2412479e+00]	 2.0696855e-01
      53	 1.2723945e+00	 1.5165747e-01	 1.3222470e+00	 1.3831306e-01	[ 1.2619760e+00]	 1.9692659e-01


.. parsed-literal::

      54	 1.2810754e+00	 1.4985424e-01	 1.3311819e+00	 1.3656997e-01	[ 1.2666828e+00]	 2.0153618e-01
      55	 1.2898392e+00	 1.4938147e-01	 1.3396250e+00	 1.3617533e-01	[ 1.2751817e+00]	 1.9905591e-01


.. parsed-literal::

      56	 1.2987970e+00	 1.4781068e-01	 1.3486578e+00	 1.3497857e-01	[ 1.2836783e+00]	 1.8347526e-01


.. parsed-literal::

      57	 1.3072599e+00	 1.4687222e-01	 1.3573536e+00	 1.3427277e-01	[ 1.2896797e+00]	 2.1822214e-01


.. parsed-literal::

      58	 1.3185738e+00	 1.4564311e-01	 1.3689369e+00	 1.3311370e-01	[ 1.2983629e+00]	 2.0874524e-01


.. parsed-literal::

      59	 1.3276043e+00	 1.4509561e-01	 1.3786253e+00	 1.3274090e-01	[ 1.3044871e+00]	 2.0815969e-01


.. parsed-literal::

      60	 1.3344996e+00	 1.4458199e-01	 1.3855535e+00	 1.3240268e-01	[ 1.3072221e+00]	 2.0756507e-01


.. parsed-literal::

      61	 1.3401497e+00	 1.4380969e-01	 1.3912984e+00	 1.3153851e-01	[ 1.3115768e+00]	 2.1069813e-01


.. parsed-literal::

      62	 1.3479619e+00	 1.4251182e-01	 1.3992778e+00	 1.3010088e-01	[ 1.3187268e+00]	 2.1226573e-01


.. parsed-literal::

      63	 1.3570290e+00	 1.4059867e-01	 1.4088061e+00	 1.2826778e-01	[ 1.3262222e+00]	 2.0775747e-01


.. parsed-literal::

      64	 1.3651085e+00	 1.3974284e-01	 1.4167319e+00	 1.2729551e-01	[ 1.3326827e+00]	 2.0704818e-01


.. parsed-literal::

      65	 1.3709767e+00	 1.3927044e-01	 1.4225334e+00	 1.2696361e-01	[ 1.3350298e+00]	 2.0855784e-01


.. parsed-literal::

      66	 1.3796582e+00	 1.3838170e-01	 1.4313201e+00	 1.2633568e-01	  1.3335866e+00 	 2.0971179e-01


.. parsed-literal::

      67	 1.3845770e+00	 1.3778494e-01	 1.4362680e+00	 1.2553967e-01	  1.3340015e+00 	 3.2047296e-01
      68	 1.3900729e+00	 1.3730367e-01	 1.4418558e+00	 1.2514541e-01	[ 1.3359609e+00]	 1.8032980e-01


.. parsed-literal::

      69	 1.3967815e+00	 1.3664365e-01	 1.4488054e+00	 1.2411446e-01	[ 1.3411736e+00]	 1.9166040e-01


.. parsed-literal::

      70	 1.4027341e+00	 1.3594242e-01	 1.4549733e+00	 1.2300294e-01	[ 1.3468283e+00]	 2.1822333e-01


.. parsed-literal::

      71	 1.4097388e+00	 1.3519299e-01	 1.4621157e+00	 1.2172578e-01	[ 1.3595681e+00]	 2.0851755e-01


.. parsed-literal::

      72	 1.4150300e+00	 1.3474954e-01	 1.4672667e+00	 1.2121737e-01	[ 1.3647619e+00]	 2.0202303e-01
      73	 1.4204957e+00	 1.3412854e-01	 1.4726794e+00	 1.2092582e-01	[ 1.3669891e+00]	 1.9019866e-01


.. parsed-literal::

      74	 1.4254007e+00	 1.3358836e-01	 1.4775431e+00	 1.2071224e-01	  1.3650425e+00 	 2.0856237e-01
      75	 1.4307373e+00	 1.3303975e-01	 1.4828629e+00	 1.2049108e-01	  1.3650443e+00 	 2.0147276e-01


.. parsed-literal::

      76	 1.4360577e+00	 1.3244646e-01	 1.4883621e+00	 1.2015513e-01	  1.3650289e+00 	 2.1129346e-01
      77	 1.4405702e+00	 1.3205309e-01	 1.4931806e+00	 1.1979711e-01	  1.3642776e+00 	 1.8042278e-01


.. parsed-literal::

      78	 1.4466392e+00	 1.3150916e-01	 1.4994659e+00	 1.1955310e-01	[ 1.3689486e+00]	 2.1329522e-01
      79	 1.4509730e+00	 1.3106871e-01	 1.5039818e+00	 1.1972254e-01	[ 1.3702347e+00]	 1.7798376e-01


.. parsed-literal::

      80	 1.4555040e+00	 1.3081902e-01	 1.5083804e+00	 1.1966621e-01	[ 1.3759069e+00]	 2.0951462e-01


.. parsed-literal::

      81	 1.4579887e+00	 1.3069846e-01	 1.5107294e+00	 1.1975923e-01	[ 1.3791180e+00]	 2.1028852e-01
      82	 1.4621480e+00	 1.3033075e-01	 1.5148151e+00	 1.1972985e-01	[ 1.3814500e+00]	 1.7382884e-01


.. parsed-literal::

      83	 1.4654944e+00	 1.3013031e-01	 1.5181647e+00	 1.2005186e-01	[ 1.3821239e+00]	 1.7319846e-01


.. parsed-literal::

      84	 1.4695887e+00	 1.2984407e-01	 1.5221679e+00	 1.1985984e-01	  1.3819223e+00 	 2.0513654e-01
      85	 1.4719361e+00	 1.2977186e-01	 1.5245476e+00	 1.1964162e-01	  1.3807116e+00 	 2.0452046e-01


.. parsed-literal::

      86	 1.4749493e+00	 1.2968303e-01	 1.5277193e+00	 1.1956456e-01	  1.3742517e+00 	 2.0958161e-01
      87	 1.4786249e+00	 1.2987767e-01	 1.5315337e+00	 1.1993438e-01	  1.3685941e+00 	 1.9970107e-01


.. parsed-literal::

      88	 1.4821252e+00	 1.2974832e-01	 1.5351454e+00	 1.2004367e-01	  1.3581615e+00 	 2.0240688e-01
      89	 1.4842065e+00	 1.2968398e-01	 1.5371403e+00	 1.2017877e-01	  1.3611761e+00 	 1.9210076e-01


.. parsed-literal::

      90	 1.4868981e+00	 1.2953542e-01	 1.5398509e+00	 1.2030246e-01	  1.3616958e+00 	 2.1671128e-01


.. parsed-literal::

      91	 1.4902765e+00	 1.2909601e-01	 1.5433220e+00	 1.2025479e-01	  1.3564663e+00 	 2.0993137e-01


.. parsed-literal::

      92	 1.4937493e+00	 1.2862949e-01	 1.5469425e+00	 1.2002370e-01	  1.3523125e+00 	 2.0957351e-01


.. parsed-literal::

      93	 1.4967217e+00	 1.2819855e-01	 1.5499235e+00	 1.1959398e-01	  1.3507454e+00 	 2.1216416e-01
      94	 1.4991612e+00	 1.2781412e-01	 1.5523279e+00	 1.1917858e-01	  1.3533809e+00 	 2.0436311e-01


.. parsed-literal::

      95	 1.5011486e+00	 1.2753270e-01	 1.5543457e+00	 1.1888702e-01	  1.3541341e+00 	 2.1428323e-01


.. parsed-literal::

      96	 1.5038170e+00	 1.2716850e-01	 1.5570058e+00	 1.1873609e-01	  1.3582281e+00 	 2.1393275e-01


.. parsed-literal::

      97	 1.5067158e+00	 1.2644159e-01	 1.5600277e+00	 1.1841647e-01	  1.3527920e+00 	 2.1174240e-01


.. parsed-literal::

      98	 1.5092366e+00	 1.2618186e-01	 1.5625719e+00	 1.1844792e-01	  1.3546087e+00 	 2.1007061e-01


.. parsed-literal::

      99	 1.5105208e+00	 1.2618886e-01	 1.5638557e+00	 1.1849180e-01	  1.3549814e+00 	 2.0776391e-01


.. parsed-literal::

     100	 1.5131289e+00	 1.2593366e-01	 1.5666195e+00	 1.1851933e-01	  1.3471722e+00 	 2.0792460e-01
     101	 1.5142481e+00	 1.2587597e-01	 1.5678708e+00	 1.1859055e-01	  1.3455496e+00 	 1.9874430e-01


.. parsed-literal::

     102	 1.5158762e+00	 1.2577643e-01	 1.5694014e+00	 1.1845322e-01	  1.3480205e+00 	 1.7283797e-01


.. parsed-literal::

     103	 1.5172020e+00	 1.2561939e-01	 1.5707224e+00	 1.1834715e-01	  1.3472604e+00 	 2.0889425e-01
     104	 1.5187316e+00	 1.2547167e-01	 1.5722795e+00	 1.1825119e-01	  1.3456058e+00 	 2.0374870e-01


.. parsed-literal::

     105	 1.5212636e+00	 1.2525619e-01	 1.5748838e+00	 1.1804069e-01	  1.3425086e+00 	 2.0923114e-01


.. parsed-literal::

     106	 1.5230453e+00	 1.2515270e-01	 1.5767523e+00	 1.1805087e-01	  1.3397351e+00 	 3.3001852e-01


.. parsed-literal::

     107	 1.5256311e+00	 1.2492042e-01	 1.5794280e+00	 1.1786108e-01	  1.3339233e+00 	 2.1011972e-01


.. parsed-literal::

     108	 1.5270537e+00	 1.2486254e-01	 1.5808456e+00	 1.1774536e-01	  1.3357754e+00 	 2.1205592e-01


.. parsed-literal::

     109	 1.5285666e+00	 1.2475206e-01	 1.5823780e+00	 1.1762917e-01	  1.3365796e+00 	 2.1449494e-01
     110	 1.5293456e+00	 1.2462132e-01	 1.5832131e+00	 1.1743043e-01	  1.3308534e+00 	 1.9775224e-01


.. parsed-literal::

     111	 1.5306753e+00	 1.2455109e-01	 1.5845198e+00	 1.1741653e-01	  1.3330950e+00 	 2.0754886e-01
     112	 1.5317633e+00	 1.2445230e-01	 1.5856335e+00	 1.1739338e-01	  1.3318721e+00 	 1.8222523e-01


.. parsed-literal::

     113	 1.5326589e+00	 1.2439904e-01	 1.5865476e+00	 1.1735224e-01	  1.3300469e+00 	 2.0622206e-01
     114	 1.5342963e+00	 1.2419353e-01	 1.5883520e+00	 1.1710549e-01	  1.3203906e+00 	 1.9533610e-01


.. parsed-literal::

     115	 1.5359466e+00	 1.2423737e-01	 1.5899932e+00	 1.1711392e-01	  1.3144130e+00 	 2.0624804e-01
     116	 1.5368363e+00	 1.2419764e-01	 1.5908613e+00	 1.1705940e-01	  1.3145977e+00 	 1.8250537e-01


.. parsed-literal::

     117	 1.5383694e+00	 1.2410628e-01	 1.5924094e+00	 1.1689610e-01	  1.3121567e+00 	 2.1947145e-01
     118	 1.5397955e+00	 1.2401655e-01	 1.5938890e+00	 1.1681915e-01	  1.3037649e+00 	 1.9927454e-01


.. parsed-literal::

     119	 1.5413739e+00	 1.2395137e-01	 1.5954819e+00	 1.1676270e-01	  1.2963834e+00 	 2.0725036e-01


.. parsed-literal::

     120	 1.5429876e+00	 1.2389415e-01	 1.5971216e+00	 1.1677690e-01	  1.2885643e+00 	 2.0367885e-01


.. parsed-literal::

     121	 1.5442782e+00	 1.2384322e-01	 1.5984545e+00	 1.1686200e-01	  1.2770169e+00 	 2.1553516e-01
     122	 1.5456431e+00	 1.2379341e-01	 1.5998274e+00	 1.1688007e-01	  1.2736483e+00 	 1.7702699e-01


.. parsed-literal::

     123	 1.5470197e+00	 1.2371598e-01	 1.6012163e+00	 1.1680899e-01	  1.2709506e+00 	 2.0564675e-01


.. parsed-literal::

     124	 1.5481186e+00	 1.2363254e-01	 1.6023921e+00	 1.1661301e-01	  1.2620016e+00 	 2.0786405e-01
     125	 1.5496632e+00	 1.2355410e-01	 1.6038972e+00	 1.1641058e-01	  1.2610123e+00 	 1.8100643e-01


.. parsed-literal::

     126	 1.5507136e+00	 1.2349498e-01	 1.6049529e+00	 1.1618835e-01	  1.2548985e+00 	 1.9508839e-01
     127	 1.5517851e+00	 1.2346529e-01	 1.6060362e+00	 1.1595783e-01	  1.2458148e+00 	 1.7269039e-01


.. parsed-literal::

     128	 1.5525148e+00	 1.2344431e-01	 1.6068208e+00	 1.1553959e-01	  1.2315527e+00 	 2.0200634e-01


.. parsed-literal::

     129	 1.5542287e+00	 1.2347481e-01	 1.6085006e+00	 1.1547752e-01	  1.2262032e+00 	 2.0492887e-01


.. parsed-literal::

     130	 1.5549139e+00	 1.2349209e-01	 1.6091681e+00	 1.1552123e-01	  1.2286547e+00 	 2.2165704e-01


.. parsed-literal::

     131	 1.5557205e+00	 1.2351242e-01	 1.6099958e+00	 1.1547863e-01	  1.2274471e+00 	 2.0874047e-01
     132	 1.5567772e+00	 1.2352483e-01	 1.6111233e+00	 1.1544096e-01	  1.2211019e+00 	 1.8973780e-01


.. parsed-literal::

     133	 1.5580133e+00	 1.2354682e-01	 1.6123928e+00	 1.1533257e-01	  1.2148081e+00 	 2.0342064e-01
     134	 1.5588219e+00	 1.2352030e-01	 1.6132335e+00	 1.1524270e-01	  1.2068850e+00 	 1.7360091e-01


.. parsed-literal::

     135	 1.5596177e+00	 1.2349493e-01	 1.6140653e+00	 1.1524172e-01	  1.1972627e+00 	 2.0379114e-01
     136	 1.5605553e+00	 1.2346366e-01	 1.6150520e+00	 1.1531108e-01	  1.1833801e+00 	 1.8266487e-01


.. parsed-literal::

     137	 1.5617382e+00	 1.2345582e-01	 1.6162644e+00	 1.1550885e-01	  1.1673854e+00 	 2.0511532e-01


.. parsed-literal::

     138	 1.5629759e+00	 1.2343883e-01	 1.6174879e+00	 1.1573653e-01	  1.1598225e+00 	 2.1814537e-01


.. parsed-literal::

     139	 1.5643779e+00	 1.2337822e-01	 1.6189153e+00	 1.1592397e-01	  1.1491082e+00 	 2.1036720e-01


.. parsed-literal::

     140	 1.5652226e+00	 1.2339391e-01	 1.6197700e+00	 1.1620965e-01	  1.1386621e+00 	 2.0848107e-01


.. parsed-literal::

     141	 1.5659735e+00	 1.2335894e-01	 1.6205204e+00	 1.1611272e-01	  1.1377812e+00 	 2.1301150e-01


.. parsed-literal::

     142	 1.5672008e+00	 1.2330052e-01	 1.6218202e+00	 1.1589841e-01	  1.1240834e+00 	 2.1340513e-01


.. parsed-literal::

     143	 1.5680714e+00	 1.2325736e-01	 1.6227292e+00	 1.1582754e-01	  1.1105050e+00 	 2.0764542e-01
     144	 1.5692731e+00	 1.2323232e-01	 1.6240150e+00	 1.1572814e-01	  1.0847279e+00 	 1.7490458e-01


.. parsed-literal::

     145	 1.5697333e+00	 1.2317762e-01	 1.6245711e+00	 1.1561567e-01	  1.0673074e+00 	 2.1104145e-01
     146	 1.5706196e+00	 1.2318135e-01	 1.6253959e+00	 1.1564200e-01	  1.0698794e+00 	 2.0106292e-01


.. parsed-literal::

     147	 1.5711463e+00	 1.2317045e-01	 1.6259088e+00	 1.1560847e-01	  1.0694053e+00 	 2.0110583e-01
     148	 1.5719348e+00	 1.2314703e-01	 1.6267205e+00	 1.1551343e-01	  1.0607616e+00 	 1.9901657e-01


.. parsed-literal::

     149	 1.5729735e+00	 1.2312765e-01	 1.6278025e+00	 1.1546329e-01	  1.0425382e+00 	 2.0792699e-01
     150	 1.5744425e+00	 1.2306611e-01	 1.6293768e+00	 1.1550602e-01	  9.9873082e-01 	 1.8475533e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.14 s, total: 2min 3s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f908475e3b0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.78 s, sys: 37 ms, total: 1.82 s
    Wall time: 565 ms


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

