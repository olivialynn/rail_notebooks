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
       1	-3.3337532e-01	 3.1736978e-01	-3.2378956e-01	 3.3334235e-01	[-3.5506991e-01]	 4.7214484e-01


.. parsed-literal::

       2	-2.6563665e-01	 3.0830504e-01	-2.4348070e-01	 3.2248488e-01	[-2.8766429e-01]	 2.3102355e-01


.. parsed-literal::

       3	-2.2064860e-01	 2.8824117e-01	-1.7860675e-01	 2.9782976e-01	[-2.1867313e-01]	 2.7975321e-01


.. parsed-literal::

       4	-1.8783516e-01	 2.7186382e-01	-1.3979583e-01	 2.7978394e-01	[-1.8052757e-01]	 3.2953334e-01


.. parsed-literal::

       5	-1.3234462e-01	 2.5671739e-01	-9.6207925e-02	 2.6252412e-01	[-1.2896763e-01]	 2.1223521e-01


.. parsed-literal::

       6	-6.6916846e-02	 2.5083041e-01	-3.7427768e-02	 2.5729796e-01	[-6.4469933e-02]	 2.0131922e-01


.. parsed-literal::

       7	-4.7473610e-02	 2.4775015e-01	-2.3438269e-02	 2.5510164e-01	[-5.2768303e-02]	 2.0931649e-01


.. parsed-literal::

       8	-3.2943821e-02	 2.4531603e-01	-1.3925082e-02	 2.5321254e-01	[-4.6008245e-02]	 2.0772743e-01


.. parsed-literal::

       9	-2.5582610e-02	 2.4398925e-01	-8.3325635e-03	 2.5168812e-01	[-3.8687341e-02]	 2.1641636e-01
      10	-1.3343695e-02	 2.4158786e-01	 2.0606969e-03	 2.4834861e-01	[-2.6448076e-02]	 1.9894862e-01


.. parsed-literal::

      11	-7.0801417e-03	 2.4066769e-01	 6.8735035e-03	 2.4767386e-01	[-2.0410738e-02]	 2.0478678e-01


.. parsed-literal::

      12	-2.4548129e-03	 2.3972789e-01	 1.1469078e-02	 2.4658470e-01	[-1.6135113e-02]	 2.1226478e-01
      13	 1.2336423e-03	 2.3891338e-01	 1.5270076e-02	 2.4581816e-01	[-1.3359435e-02]	 1.8829203e-01


.. parsed-literal::

      14	 3.6951777e-02	 2.3251131e-01	 5.4301936e-02	 2.3858429e-01	[ 2.6832185e-02]	 2.9611397e-01


.. parsed-literal::

      15	 1.2730375e-01	 2.2393738e-01	 1.5147304e-01	 2.2675950e-01	[ 1.4575384e-01]	 2.1479702e-01


.. parsed-literal::

      16	 2.3713550e-01	 2.1497768e-01	 2.6297073e-01	 2.1733582e-01	[ 2.5490283e-01]	 2.1641278e-01


.. parsed-literal::

      17	 2.9369451e-01	 2.1094518e-01	 3.2263150e-01	 2.1328265e-01	[ 3.0942084e-01]	 3.2438111e-01
      18	 3.2705419e-01	 2.0775852e-01	 3.5741838e-01	 2.1205636e-01	[ 3.3136406e-01]	 1.9309711e-01


.. parsed-literal::

      19	 3.7090828e-01	 2.0680005e-01	 4.0250503e-01	 2.1094845e-01	[ 3.7757015e-01]	 1.9156623e-01


.. parsed-literal::

      20	 4.4172767e-01	 2.0845110e-01	 4.7516093e-01	 2.1076933e-01	[ 4.4993670e-01]	 2.0453382e-01


.. parsed-literal::

      21	 5.1792589e-01	 2.0774730e-01	 5.5357100e-01	 2.0956860e-01	[ 5.1931393e-01]	 2.1945190e-01


.. parsed-literal::

      22	 5.7444311e-01	 2.0407263e-01	 6.1218699e-01	 2.0619722e-01	[ 5.6936232e-01]	 2.0307398e-01
      23	 6.0921777e-01	 2.0058247e-01	 6.4882993e-01	 2.0238363e-01	[ 5.8168152e-01]	 2.0485473e-01


.. parsed-literal::

      24	 6.4317394e-01	 1.9848128e-01	 6.8148701e-01	 2.0021642e-01	[ 6.2349247e-01]	 2.0438409e-01


.. parsed-literal::

      25	 6.6736549e-01	 1.9625666e-01	 7.0502756e-01	 1.9947442e-01	[ 6.4589780e-01]	 2.1159339e-01


.. parsed-literal::

      26	 6.9868952e-01	 1.9506784e-01	 7.3569768e-01	 1.9989664e-01	[ 6.7070168e-01]	 3.1595635e-01


.. parsed-literal::

      27	 7.2030365e-01	 1.9789153e-01	 7.5746335e-01	 1.9994007e-01	[ 6.9510527e-01]	 2.1048021e-01


.. parsed-literal::

      28	 7.5126880e-01	 1.9518619e-01	 7.8991791e-01	 1.9753240e-01	[ 7.2419301e-01]	 2.1080136e-01
      29	 7.8281045e-01	 1.9521191e-01	 8.2231752e-01	 1.9684262e-01	[ 7.6210022e-01]	 1.8044734e-01


.. parsed-literal::

      30	 8.2575999e-01	 1.9753502e-01	 8.6646092e-01	 1.9882645e-01	[ 8.1778909e-01]	 2.1372390e-01
      31	 8.4831667e-01	 1.9907077e-01	 8.8953843e-01	 2.0115071e-01	[ 8.5222596e-01]	 1.8703246e-01


.. parsed-literal::

      32	 8.7303695e-01	 1.9578016e-01	 9.1515832e-01	 1.9776314e-01	[ 8.7385883e-01]	 2.0432663e-01


.. parsed-literal::

      33	 8.9115301e-01	 1.9154360e-01	 9.3301008e-01	 1.9368415e-01	[ 8.8742420e-01]	 2.0655346e-01
      34	 9.1400845e-01	 1.8811870e-01	 9.5588135e-01	 1.8958635e-01	[ 9.0785072e-01]	 1.9946766e-01


.. parsed-literal::

      35	 9.4075393e-01	 1.8074206e-01	 9.8309169e-01	 1.8391256e-01	[ 9.3111405e-01]	 2.1260333e-01


.. parsed-literal::

      36	 9.6041254e-01	 1.7886474e-01	 1.0041596e+00	 1.8098581e-01	[ 9.4366825e-01]	 2.0417643e-01


.. parsed-literal::

      37	 9.7534548e-01	 1.7738876e-01	 1.0192554e+00	 1.7979383e-01	[ 9.5808049e-01]	 2.0874023e-01


.. parsed-literal::

      38	 9.9495637e-01	 1.7446251e-01	 1.0396801e+00	 1.7764835e-01	[ 9.7091323e-01]	 2.1128416e-01


.. parsed-literal::

      39	 1.0067985e+00	 1.7269702e-01	 1.0525550e+00	 1.7548579e-01	[ 9.7791330e-01]	 2.0586538e-01


.. parsed-literal::

      40	 1.0187632e+00	 1.7042109e-01	 1.0648323e+00	 1.7265155e-01	[ 9.9298535e-01]	 2.6082611e-01


.. parsed-literal::

      41	 1.0259595e+00	 1.6977217e-01	 1.0715952e+00	 1.7162201e-01	[ 1.0036171e+00]	 2.1048355e-01
      42	 1.0401093e+00	 1.6657499e-01	 1.0860492e+00	 1.6905199e-01	[ 1.0183011e+00]	 1.7332363e-01


.. parsed-literal::

      43	 1.0513799e+00	 1.6472032e-01	 1.0980638e+00	 1.6788872e-01	[ 1.0268469e+00]	 1.8454123e-01
      44	 1.0640100e+00	 1.6184398e-01	 1.1109778e+00	 1.6687980e-01	[ 1.0387845e+00]	 1.7957354e-01


.. parsed-literal::

      45	 1.0744515e+00	 1.5932265e-01	 1.1218321e+00	 1.6707046e-01	[ 1.0469025e+00]	 2.1113038e-01
      46	 1.0863135e+00	 1.5786794e-01	 1.1338565e+00	 1.6824542e-01	[ 1.0553767e+00]	 1.7399573e-01


.. parsed-literal::

      47	 1.0958257e+00	 1.5645486e-01	 1.1440086e+00	 1.7471088e-01	  1.0533763e+00 	 2.0846653e-01


.. parsed-literal::

      48	 1.1108276e+00	 1.5611860e-01	 1.1585876e+00	 1.7212238e-01	[ 1.0689582e+00]	 2.0916700e-01


.. parsed-literal::

      49	 1.1183306e+00	 1.5533241e-01	 1.1663364e+00	 1.7126295e-01	[ 1.0753738e+00]	 2.0976281e-01
      50	 1.1280891e+00	 1.5394557e-01	 1.1766255e+00	 1.7005978e-01	[ 1.0801483e+00]	 1.9876695e-01


.. parsed-literal::

      51	 1.1397820e+00	 1.5287076e-01	 1.1887096e+00	 1.7060935e-01	[ 1.0835202e+00]	 1.9610786e-01


.. parsed-literal::

      52	 1.1489700e+00	 1.5150701e-01	 1.1986958e+00	 1.7062974e-01	  1.0794538e+00 	 2.1413732e-01
      53	 1.1601027e+00	 1.5094390e-01	 1.2096194e+00	 1.6972710e-01	[ 1.0876915e+00]	 1.9044828e-01


.. parsed-literal::

      54	 1.1728447e+00	 1.5003592e-01	 1.2221289e+00	 1.6647190e-01	[ 1.0955634e+00]	 2.0679545e-01


.. parsed-literal::

      55	 1.1836769e+00	 1.4944154e-01	 1.2328787e+00	 1.6535274e-01	[ 1.1055011e+00]	 2.1189809e-01


.. parsed-literal::

      56	 1.1973475e+00	 1.4702933e-01	 1.2470209e+00	 1.5991270e-01	[ 1.1123723e+00]	 2.0611525e-01


.. parsed-literal::

      57	 1.2069425e+00	 1.4558313e-01	 1.2565856e+00	 1.5667521e-01	[ 1.1231955e+00]	 2.0965791e-01
      58	 1.2154978e+00	 1.4528005e-01	 1.2649453e+00	 1.5577961e-01	[ 1.1347897e+00]	 2.0342708e-01


.. parsed-literal::

      59	 1.2275130e+00	 1.4369769e-01	 1.2771996e+00	 1.5307625e-01	[ 1.1454497e+00]	 2.0313168e-01
      60	 1.2383490e+00	 1.4221670e-01	 1.2881896e+00	 1.5063382e-01	[ 1.1562889e+00]	 1.7639852e-01


.. parsed-literal::

      61	 1.2472679e+00	 1.3925155e-01	 1.2977614e+00	 1.4761310e-01	  1.1527070e+00 	 2.0008707e-01


.. parsed-literal::

      62	 1.2619107e+00	 1.3933626e-01	 1.3120532e+00	 1.4738379e-01	[ 1.1642149e+00]	 2.0559692e-01
      63	 1.2676293e+00	 1.3901011e-01	 1.3177080e+00	 1.4725295e-01	[ 1.1687117e+00]	 1.7305040e-01


.. parsed-literal::

      64	 1.2805002e+00	 1.3769682e-01	 1.3308297e+00	 1.4625198e-01	[ 1.1703570e+00]	 2.0819807e-01
      65	 1.2876569e+00	 1.3690213e-01	 1.3383126e+00	 1.4631760e-01	[ 1.1736572e+00]	 1.6600871e-01


.. parsed-literal::

      66	 1.2969282e+00	 1.3550052e-01	 1.3476816e+00	 1.4496689e-01	[ 1.1784234e+00]	 2.0890021e-01
      67	 1.3039846e+00	 1.3569015e-01	 1.3549384e+00	 1.4358882e-01	  1.1720475e+00 	 1.8249917e-01


.. parsed-literal::

      68	 1.3095219e+00	 1.3515014e-01	 1.3605964e+00	 1.4350936e-01	  1.1740497e+00 	 2.0974803e-01
      69	 1.3173677e+00	 1.3449863e-01	 1.3689522e+00	 1.4368965e-01	  1.1620475e+00 	 1.8560386e-01


.. parsed-literal::

      70	 1.3248602e+00	 1.3381416e-01	 1.3765826e+00	 1.4403223e-01	  1.1616099e+00 	 1.7332149e-01


.. parsed-literal::

      71	 1.3330318e+00	 1.3329845e-01	 1.3848467e+00	 1.4372431e-01	  1.1520461e+00 	 2.1429706e-01
      72	 1.3398094e+00	 1.3258486e-01	 1.3917891e+00	 1.4336565e-01	  1.1529439e+00 	 1.8679523e-01


.. parsed-literal::

      73	 1.3462284e+00	 1.3219236e-01	 1.3982872e+00	 1.4239111e-01	  1.1474830e+00 	 1.8883824e-01


.. parsed-literal::

      74	 1.3521832e+00	 1.3163042e-01	 1.4042305e+00	 1.4174104e-01	  1.1492201e+00 	 2.0736432e-01
      75	 1.3595356e+00	 1.3101916e-01	 1.4117829e+00	 1.4129723e-01	  1.1451235e+00 	 1.8062782e-01


.. parsed-literal::

      76	 1.3658022e+00	 1.3083784e-01	 1.4181182e+00	 1.4125996e-01	  1.1511828e+00 	 2.0695949e-01
      77	 1.3718575e+00	 1.3074175e-01	 1.4242933e+00	 1.4213056e-01	  1.1588193e+00 	 2.0414829e-01


.. parsed-literal::

      78	 1.3761045e+00	 1.3049847e-01	 1.4285158e+00	 1.4198850e-01	  1.1730648e+00 	 1.9978070e-01


.. parsed-literal::

      79	 1.3816272e+00	 1.3006412e-01	 1.4343364e+00	 1.4176383e-01	[ 1.1840354e+00]	 2.1342492e-01


.. parsed-literal::

      80	 1.3864884e+00	 1.2969178e-01	 1.4393143e+00	 1.4160639e-01	[ 1.1990869e+00]	 2.0725918e-01
      81	 1.3913910e+00	 1.2947806e-01	 1.4441663e+00	 1.4157842e-01	[ 1.2102001e+00]	 1.9613695e-01


.. parsed-literal::

      82	 1.3958678e+00	 1.2932775e-01	 1.4486301e+00	 1.4122951e-01	[ 1.2160158e+00]	 2.1720672e-01
      83	 1.3989014e+00	 1.2891521e-01	 1.4517269e+00	 1.4053028e-01	[ 1.2231195e+00]	 1.7302918e-01


.. parsed-literal::

      84	 1.4022538e+00	 1.2875053e-01	 1.4551229e+00	 1.4012407e-01	[ 1.2262499e+00]	 2.0711279e-01


.. parsed-literal::

      85	 1.4070079e+00	 1.2856749e-01	 1.4600259e+00	 1.3932212e-01	[ 1.2286482e+00]	 2.1667814e-01
      86	 1.4104055e+00	 1.2840926e-01	 1.4634895e+00	 1.3922725e-01	  1.2279195e+00 	 1.9570041e-01


.. parsed-literal::

      87	 1.4150874e+00	 1.2821716e-01	 1.4682636e+00	 1.3935737e-01	  1.2255620e+00 	 2.1058631e-01
      88	 1.4187449e+00	 1.2770756e-01	 1.4720028e+00	 1.3878251e-01	  1.2202649e+00 	 1.7464924e-01


.. parsed-literal::

      89	 1.4218613e+00	 1.2748640e-01	 1.4751681e+00	 1.3864342e-01	  1.2243642e+00 	 1.8941712e-01
      90	 1.4270280e+00	 1.2685912e-01	 1.4805030e+00	 1.3800579e-01	  1.2260865e+00 	 1.9221997e-01


.. parsed-literal::

      91	 1.4301851e+00	 1.2648409e-01	 1.4837583e+00	 1.3676510e-01	[ 1.2375975e+00]	 2.1500087e-01


.. parsed-literal::

      92	 1.4330347e+00	 1.2633889e-01	 1.4865052e+00	 1.3656073e-01	[ 1.2385042e+00]	 2.1044874e-01


.. parsed-literal::

      93	 1.4355210e+00	 1.2611787e-01	 1.4889562e+00	 1.3607820e-01	[ 1.2388650e+00]	 2.0461249e-01


.. parsed-literal::

      94	 1.4386382e+00	 1.2594634e-01	 1.4920241e+00	 1.3573304e-01	[ 1.2399926e+00]	 2.0418429e-01


.. parsed-literal::

      95	 1.4417187e+00	 1.2554736e-01	 1.4952441e+00	 1.3537530e-01	  1.2396075e+00 	 2.0557261e-01


.. parsed-literal::

      96	 1.4456620e+00	 1.2544376e-01	 1.4991295e+00	 1.3533985e-01	[ 1.2438890e+00]	 2.0955777e-01


.. parsed-literal::

      97	 1.4476849e+00	 1.2532057e-01	 1.5011945e+00	 1.3540732e-01	  1.2432599e+00 	 2.0604181e-01


.. parsed-literal::

      98	 1.4507177e+00	 1.2502642e-01	 1.5042955e+00	 1.3522048e-01	[ 1.2446281e+00]	 2.0409012e-01
      99	 1.4542359e+00	 1.2474847e-01	 1.5078656e+00	 1.3495300e-01	[ 1.2480754e+00]	 1.7257500e-01


.. parsed-literal::

     100	 1.4574670e+00	 1.2458435e-01	 1.5111245e+00	 1.3502060e-01	[ 1.2498299e+00]	 2.0831513e-01


.. parsed-literal::

     101	 1.4597702e+00	 1.2460841e-01	 1.5133844e+00	 1.3514792e-01	[ 1.2515631e+00]	 2.0950079e-01


.. parsed-literal::

     102	 1.4617768e+00	 1.2463446e-01	 1.5153772e+00	 1.3524247e-01	  1.2503634e+00 	 2.0596290e-01


.. parsed-literal::

     103	 1.4642106e+00	 1.2464677e-01	 1.5179014e+00	 1.3548383e-01	  1.2467924e+00 	 2.1559405e-01
     104	 1.4674304e+00	 1.2451211e-01	 1.5212853e+00	 1.3558070e-01	  1.2439189e+00 	 1.8628573e-01


.. parsed-literal::

     105	 1.4700624e+00	 1.2406947e-01	 1.5240740e+00	 1.3511402e-01	  1.2378228e+00 	 2.0069861e-01
     106	 1.4727806e+00	 1.2380407e-01	 1.5267520e+00	 1.3496901e-01	  1.2450165e+00 	 1.9989538e-01


.. parsed-literal::

     107	 1.4746911e+00	 1.2352986e-01	 1.5285859e+00	 1.3485572e-01	  1.2506300e+00 	 2.1066928e-01


.. parsed-literal::

     108	 1.4768847e+00	 1.2333052e-01	 1.5307732e+00	 1.3496097e-01	[ 1.2561969e+00]	 2.0800257e-01


.. parsed-literal::

     109	 1.4789917e+00	 1.2306161e-01	 1.5329722e+00	 1.3535545e-01	  1.2553350e+00 	 2.0568371e-01


.. parsed-literal::

     110	 1.4809547e+00	 1.2312182e-01	 1.5348786e+00	 1.3527370e-01	[ 1.2564110e+00]	 2.1675134e-01


.. parsed-literal::

     111	 1.4822339e+00	 1.2314775e-01	 1.5362229e+00	 1.3532746e-01	  1.2537339e+00 	 2.0938230e-01


.. parsed-literal::

     112	 1.4843158e+00	 1.2310097e-01	 1.5384004e+00	 1.3519733e-01	  1.2492489e+00 	 2.0475173e-01
     113	 1.4866577e+00	 1.2298183e-01	 1.5409113e+00	 1.3530205e-01	  1.2440815e+00 	 1.8704939e-01


.. parsed-literal::

     114	 1.4891384e+00	 1.2282792e-01	 1.5434522e+00	 1.3549172e-01	  1.2460914e+00 	 2.1244383e-01
     115	 1.4919321e+00	 1.2266265e-01	 1.5462450e+00	 1.3567724e-01	  1.2502147e+00 	 1.9171429e-01


.. parsed-literal::

     116	 1.4942894e+00	 1.2257107e-01	 1.5486074e+00	 1.3621628e-01	[ 1.2592135e+00]	 2.1451259e-01


.. parsed-literal::

     117	 1.4964546e+00	 1.2255230e-01	 1.5507524e+00	 1.3645470e-01	[ 1.2605932e+00]	 2.1431589e-01


.. parsed-literal::

     118	 1.4983056e+00	 1.2252496e-01	 1.5525818e+00	 1.3647353e-01	[ 1.2621973e+00]	 2.0646524e-01


.. parsed-literal::

     119	 1.5005259e+00	 1.2242570e-01	 1.5549006e+00	 1.3650414e-01	[ 1.2636092e+00]	 2.0506239e-01


.. parsed-literal::

     120	 1.5026632e+00	 1.2234131e-01	 1.5571540e+00	 1.3645498e-01	[ 1.2695200e+00]	 2.1837258e-01


.. parsed-literal::

     121	 1.5043004e+00	 1.2221943e-01	 1.5587836e+00	 1.3608238e-01	  1.2681642e+00 	 2.1222162e-01


.. parsed-literal::

     122	 1.5058237e+00	 1.2211365e-01	 1.5603694e+00	 1.3597098e-01	  1.2688218e+00 	 2.1423030e-01
     123	 1.5081124e+00	 1.2197896e-01	 1.5627143e+00	 1.3572355e-01	  1.2694067e+00 	 1.9716215e-01


.. parsed-literal::

     124	 1.5093256e+00	 1.2184643e-01	 1.5641438e+00	 1.3544736e-01	  1.2689936e+00 	 2.1169949e-01


.. parsed-literal::

     125	 1.5123684e+00	 1.2177769e-01	 1.5669815e+00	 1.3532629e-01	[ 1.2727182e+00]	 2.0826697e-01
     126	 1.5133118e+00	 1.2175824e-01	 1.5678614e+00	 1.3529590e-01	[ 1.2742855e+00]	 1.8746209e-01


.. parsed-literal::

     127	 1.5152959e+00	 1.2170637e-01	 1.5697905e+00	 1.3520536e-01	[ 1.2773468e+00]	 2.1921277e-01
     128	 1.5169697e+00	 1.2163331e-01	 1.5715221e+00	 1.3514348e-01	[ 1.2782512e+00]	 2.0145559e-01


.. parsed-literal::

     129	 1.5192505e+00	 1.2158567e-01	 1.5738231e+00	 1.3507000e-01	[ 1.2784243e+00]	 2.0780659e-01


.. parsed-literal::

     130	 1.5206765e+00	 1.2149180e-01	 1.5752898e+00	 1.3501864e-01	  1.2766801e+00 	 2.0456624e-01
     131	 1.5219036e+00	 1.2136864e-01	 1.5766093e+00	 1.3495951e-01	  1.2726083e+00 	 2.0053840e-01


.. parsed-literal::

     132	 1.5233806e+00	 1.2122926e-01	 1.5781783e+00	 1.3508976e-01	  1.2700030e+00 	 1.7437959e-01
     133	 1.5251067e+00	 1.2105089e-01	 1.5799325e+00	 1.3496031e-01	  1.2627075e+00 	 1.8804121e-01


.. parsed-literal::

     134	 1.5264919e+00	 1.2095304e-01	 1.5813064e+00	 1.3492163e-01	  1.2597286e+00 	 2.0987344e-01
     135	 1.5279056e+00	 1.2083022e-01	 1.5826784e+00	 1.3483441e-01	  1.2577803e+00 	 2.0520544e-01


.. parsed-literal::

     136	 1.5295117e+00	 1.2065448e-01	 1.5842724e+00	 1.3482551e-01	  1.2530732e+00 	 2.1361971e-01


.. parsed-literal::

     137	 1.5311172e+00	 1.2056261e-01	 1.5858615e+00	 1.3474351e-01	  1.2531309e+00 	 2.1192741e-01


.. parsed-literal::

     138	 1.5324666e+00	 1.2045959e-01	 1.5872641e+00	 1.3469246e-01	  1.2517954e+00 	 2.0873690e-01


.. parsed-literal::

     139	 1.5337249e+00	 1.2037074e-01	 1.5886141e+00	 1.3467995e-01	  1.2486217e+00 	 2.1171403e-01


.. parsed-literal::

     140	 1.5348572e+00	 1.2020207e-01	 1.5900582e+00	 1.3484973e-01	  1.2369184e+00 	 2.1277976e-01


.. parsed-literal::

     141	 1.5370738e+00	 1.2015552e-01	 1.5922220e+00	 1.3476954e-01	  1.2330128e+00 	 2.0923758e-01
     142	 1.5378688e+00	 1.2013107e-01	 1.5929871e+00	 1.3476226e-01	  1.2308332e+00 	 2.0674276e-01


.. parsed-literal::

     143	 1.5393384e+00	 1.2003112e-01	 1.5944495e+00	 1.3476394e-01	  1.2230460e+00 	 1.9430470e-01


.. parsed-literal::

     144	 1.5412279e+00	 1.1983682e-01	 1.5963455e+00	 1.3485431e-01	  1.2109180e+00 	 2.1283293e-01


.. parsed-literal::

     145	 1.5429315e+00	 1.1959418e-01	 1.5980754e+00	 1.3486579e-01	  1.1892796e+00 	 2.1051407e-01


.. parsed-literal::

     146	 1.5442885e+00	 1.1950218e-01	 1.5994106e+00	 1.3496542e-01	  1.1897293e+00 	 2.0940924e-01


.. parsed-literal::

     147	 1.5455235e+00	 1.1937986e-01	 1.6006487e+00	 1.3500283e-01	  1.1886631e+00 	 2.1412778e-01
     148	 1.5466084e+00	 1.1927749e-01	 1.6017650e+00	 1.3511850e-01	  1.1868386e+00 	 1.8862224e-01


.. parsed-literal::

     149	 1.5484557e+00	 1.1910878e-01	 1.6036898e+00	 1.3517703e-01	  1.1812064e+00 	 2.1273518e-01


.. parsed-literal::

     150	 1.5499322e+00	 1.1897344e-01	 1.6052409e+00	 1.3527674e-01	  1.1733233e+00 	 2.1219373e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.14 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0c9407e6b0>



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
    CPU times: user 2.13 s, sys: 69.9 ms, total: 2.2 s
    Wall time: 680 ms


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

