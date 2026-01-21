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
       1	-3.4661060e-01	 3.2152470e-01	-3.3694653e-01	 3.1762326e-01	[-3.2891596e-01]	 5.9391117e-01


.. parsed-literal::

       2	-2.7632665e-01	 3.1115099e-01	-2.5277489e-01	 3.0743476e-01	[-2.4151872e-01]	 2.9748869e-01


.. parsed-literal::

       3	-2.3198160e-01	 2.9058611e-01	-1.9043456e-01	 2.8746874e-01	[-1.7866708e-01]	 3.3956861e-01


.. parsed-literal::

       4	-1.9730893e-01	 2.6636049e-01	-1.5482990e-01	 2.6759114e-01	[-1.5942727e-01]	 2.1601963e-01


.. parsed-literal::

       5	-1.0851187e-01	 2.5762728e-01	-7.3855233e-02	 2.5957368e-01	[-8.4349784e-02]	 2.1095538e-01


.. parsed-literal::

       6	-6.9756918e-02	 2.5113383e-01	-3.8418961e-02	 2.5567475e-01	[-5.4786266e-02]	 2.2483802e-01


.. parsed-literal::

       7	-5.1528660e-02	 2.4843084e-01	-2.6607513e-02	 2.5316626e-01	[-4.5279388e-02]	 2.1325612e-01


.. parsed-literal::

       8	-3.5008238e-02	 2.4552702e-01	-1.4461158e-02	 2.5097706e-01	[-3.6818882e-02]	 2.3119688e-01


.. parsed-literal::

       9	-2.0375899e-02	 2.4278981e-01	-2.8831492e-03	 2.4916789e-01	[-3.0268254e-02]	 2.1124578e-01


.. parsed-literal::

      10	-1.1192486e-02	 2.4109719e-01	 4.3156605e-03	 2.4678678e-01	[-1.8691931e-02]	 2.1780205e-01


.. parsed-literal::

      11	-4.6052912e-03	 2.3991990e-01	 9.9098287e-03	 2.4629898e-01	[-1.8100644e-02]	 2.1620274e-01


.. parsed-literal::

      12	-1.6576755e-03	 2.3946394e-01	 1.2549442e-02	 2.4620151e-01	[-1.5864241e-02]	 2.2384548e-01


.. parsed-literal::

      13	 1.8819564e-03	 2.3880388e-01	 1.5924644e-02	 2.4570250e-01	[-1.3169460e-02]	 2.2578812e-01


.. parsed-literal::

      14	 4.1286990e-02	 2.3098363e-01	 5.7079779e-02	 2.4181225e-01	[ 2.2112075e-02]	 3.6433601e-01


.. parsed-literal::

      15	 6.8802358e-02	 2.2528526e-01	 8.6428633e-02	 2.3432557e-01	[ 6.2532642e-02]	 3.8375425e-01


.. parsed-literal::

      16	 1.2370347e-01	 2.1971553e-01	 1.4412005e-01	 2.2901203e-01	[ 1.2310424e-01]	 2.1774435e-01


.. parsed-literal::

      17	 2.4802237e-01	 2.1790277e-01	 2.7584688e-01	 2.2900889e-01	[ 2.4865948e-01]	 2.2329879e-01


.. parsed-literal::

      18	 2.8988837e-01	 2.1740148e-01	 3.2029487e-01	 2.2879398e-01	[ 2.9766871e-01]	 2.1943545e-01


.. parsed-literal::

      19	 3.3702555e-01	 2.1588953e-01	 3.6764491e-01	 2.2317472e-01	[ 3.5091665e-01]	 2.1219850e-01


.. parsed-literal::

      20	 3.9155598e-01	 2.1066990e-01	 4.2335379e-01	 2.1639567e-01	[ 4.0913138e-01]	 2.3334789e-01


.. parsed-literal::

      21	 4.5206822e-01	 2.0681852e-01	 4.8537550e-01	 2.1542093e-01	[ 4.5875304e-01]	 2.1792173e-01


.. parsed-literal::

      22	 5.3389327e-01	 2.0505248e-01	 5.7098730e-01	 2.1583434e-01	[ 5.3140232e-01]	 2.1316195e-01


.. parsed-literal::

      23	 5.8613700e-01	 2.0317041e-01	 6.2480291e-01	 2.1241261e-01	[ 5.9708474e-01]	 2.2695875e-01


.. parsed-literal::

      24	 6.2048200e-01	 2.0046633e-01	 6.5838358e-01	 2.0929818e-01	[ 6.3597365e-01]	 2.3036361e-01


.. parsed-literal::

      25	 6.4973615e-01	 1.9939949e-01	 6.8812972e-01	 2.0732388e-01	[ 6.6605500e-01]	 2.2647834e-01


.. parsed-literal::

      26	 6.9893146e-01	 2.0026655e-01	 7.3670601e-01	 2.0928619e-01	[ 7.1684421e-01]	 2.1504498e-01


.. parsed-literal::

      27	 7.1387844e-01	 2.0896789e-01	 7.5167998e-01	 2.1612450e-01	[ 7.3801569e-01]	 2.3271441e-01


.. parsed-literal::

      28	 7.4656120e-01	 2.0690300e-01	 7.8346121e-01	 2.1461277e-01	[ 7.7695626e-01]	 2.2780418e-01


.. parsed-literal::

      29	 7.7277900e-01	 2.0654432e-01	 8.1175300e-01	 2.1437180e-01	[ 8.0373385e-01]	 2.2505474e-01


.. parsed-literal::

      30	 7.9896908e-01	 2.0217267e-01	 8.3841339e-01	 2.1173937e-01	[ 8.2613687e-01]	 2.2305465e-01


.. parsed-literal::

      31	 8.3344585e-01	 1.9453906e-01	 8.7472714e-01	 2.0595354e-01	[ 8.5490438e-01]	 2.1461248e-01


.. parsed-literal::

      32	 8.5347190e-01	 1.9659331e-01	 8.9405075e-01	 2.0940817e-01	[ 8.6803129e-01]	 2.2814131e-01


.. parsed-literal::

      33	 8.7417242e-01	 1.9303219e-01	 9.1481468e-01	 2.0633809e-01	[ 8.9024683e-01]	 2.3682618e-01


.. parsed-literal::

      34	 8.8913938e-01	 1.9177637e-01	 9.3020096e-01	 2.0523886e-01	[ 9.0107735e-01]	 2.3067069e-01


.. parsed-literal::

      35	 9.0456885e-01	 1.8934777e-01	 9.4637608e-01	 2.0216219e-01	[ 9.1116965e-01]	 2.3108840e-01


.. parsed-literal::

      36	 9.2940237e-01	 1.8707744e-01	 9.7257416e-01	 1.9929480e-01	[ 9.2934620e-01]	 2.3492026e-01


.. parsed-literal::

      37	 9.5222450e-01	 1.8441232e-01	 9.9642595e-01	 1.9629222e-01	[ 9.4823071e-01]	 2.2531724e-01


.. parsed-literal::

      38	 9.7239755e-01	 1.8351444e-01	 1.0171284e+00	 1.9680273e-01	[ 9.6902673e-01]	 2.3567939e-01


.. parsed-literal::

      39	 9.8497419e-01	 1.8383321e-01	 1.0305779e+00	 1.9789431e-01	[ 9.7807531e-01]	 2.2762895e-01


.. parsed-literal::

      40	 9.9497837e-01	 1.8378370e-01	 1.0411125e+00	 1.9882156e-01	[ 9.8557870e-01]	 2.1767068e-01


.. parsed-literal::

      41	 1.0043423e+00	 1.8430971e-01	 1.0505589e+00	 1.9943606e-01	[ 9.9385017e-01]	 2.2935486e-01


.. parsed-literal::

      42	 1.0174946e+00	 1.8463727e-01	 1.0643840e+00	 1.9979126e-01	[ 9.9998376e-01]	 2.3046255e-01


.. parsed-literal::

      43	 1.0302674e+00	 1.8303593e-01	 1.0774449e+00	 1.9783989e-01	[ 1.0061278e+00]	 2.0782971e-01


.. parsed-literal::

      44	 1.0416111e+00	 1.8128517e-01	 1.0886141e+00	 1.9595588e-01	[ 1.0107337e+00]	 2.3343563e-01


.. parsed-literal::

      45	 1.0578451e+00	 1.7876038e-01	 1.1051875e+00	 1.9281321e-01	[ 1.0271154e+00]	 2.2660041e-01


.. parsed-literal::

      46	 1.0712577e+00	 1.7497798e-01	 1.1186471e+00	 1.8960389e-01	[ 1.0336798e+00]	 2.0614505e-01


.. parsed-literal::

      47	 1.0817230e+00	 1.7430912e-01	 1.1289041e+00	 1.8826048e-01	[ 1.0509917e+00]	 2.4032855e-01


.. parsed-literal::

      48	 1.0940248e+00	 1.7370403e-01	 1.1415452e+00	 1.8761373e-01	[ 1.0671991e+00]	 2.2450113e-01


.. parsed-literal::

      49	 1.1041333e+00	 1.7264325e-01	 1.1520865e+00	 1.8580979e-01	[ 1.0726198e+00]	 2.1109152e-01


.. parsed-literal::

      50	 1.1188785e+00	 1.7050889e-01	 1.1670505e+00	 1.8418150e-01	[ 1.0851571e+00]	 2.1081614e-01


.. parsed-literal::

      51	 1.1340795e+00	 1.6709570e-01	 1.1826812e+00	 1.8138979e-01	[ 1.0868700e+00]	 2.1058488e-01


.. parsed-literal::

      52	 1.1495863e+00	 1.6408284e-01	 1.1987489e+00	 1.7943060e-01	[ 1.0926587e+00]	 2.3608875e-01


.. parsed-literal::

      53	 1.1603029e+00	 1.6078230e-01	 1.2097735e+00	 1.7727574e-01	[ 1.0936608e+00]	 2.1466184e-01


.. parsed-literal::

      54	 1.1690877e+00	 1.6035347e-01	 1.2187074e+00	 1.7666018e-01	[ 1.1028810e+00]	 2.3033834e-01


.. parsed-literal::

      55	 1.1791954e+00	 1.5951670e-01	 1.2290552e+00	 1.7586057e-01	[ 1.1145359e+00]	 2.1975112e-01


.. parsed-literal::

      56	 1.1906228e+00	 1.5758942e-01	 1.2411117e+00	 1.7402706e-01	[ 1.1219198e+00]	 2.3034573e-01


.. parsed-literal::

      57	 1.2010133e+00	 1.5541537e-01	 1.2516565e+00	 1.7179622e-01	[ 1.1385363e+00]	 2.0772314e-01


.. parsed-literal::

      58	 1.2106733e+00	 1.5414135e-01	 1.2615394e+00	 1.7053945e-01	[ 1.1474304e+00]	 2.2597575e-01


.. parsed-literal::

      59	 1.2169870e+00	 1.5342706e-01	 1.2678316e+00	 1.6985525e-01	[ 1.1502898e+00]	 2.0665359e-01


.. parsed-literal::

      60	 1.2287505e+00	 1.5169543e-01	 1.2801172e+00	 1.6823805e-01	  1.1486204e+00 	 2.0666027e-01


.. parsed-literal::

      61	 1.2343844e+00	 1.5081682e-01	 1.2856271e+00	 1.6686855e-01	[ 1.1586465e+00]	 2.3138285e-01


.. parsed-literal::

      62	 1.2443455e+00	 1.4958218e-01	 1.2953107e+00	 1.6554014e-01	[ 1.1704937e+00]	 2.2116876e-01


.. parsed-literal::

      63	 1.2515025e+00	 1.4907502e-01	 1.3025847e+00	 1.6477648e-01	[ 1.1787008e+00]	 2.1321130e-01


.. parsed-literal::

      64	 1.2609932e+00	 1.4867350e-01	 1.3122920e+00	 1.6384044e-01	[ 1.1864776e+00]	 2.2995400e-01


.. parsed-literal::

      65	 1.2718845e+00	 1.4779596e-01	 1.3232229e+00	 1.6207924e-01	[ 1.2031243e+00]	 2.2502422e-01


.. parsed-literal::

      66	 1.2818341e+00	 1.4683952e-01	 1.3337587e+00	 1.6021938e-01	[ 1.2081218e+00]	 2.3626852e-01


.. parsed-literal::

      67	 1.2890341e+00	 1.4634697e-01	 1.3410390e+00	 1.5972457e-01	[ 1.2133697e+00]	 2.1505570e-01


.. parsed-literal::

      68	 1.2975286e+00	 1.4434612e-01	 1.3497722e+00	 1.5801377e-01	[ 1.2160128e+00]	 2.2921991e-01


.. parsed-literal::

      69	 1.3050549e+00	 1.4370439e-01	 1.3572238e+00	 1.5739707e-01	[ 1.2238507e+00]	 2.3285890e-01


.. parsed-literal::

      70	 1.3112753e+00	 1.4284162e-01	 1.3633628e+00	 1.5652015e-01	[ 1.2286997e+00]	 2.2531438e-01


.. parsed-literal::

      71	 1.3222993e+00	 1.4123643e-01	 1.3744701e+00	 1.5487817e-01	[ 1.2331203e+00]	 2.1672440e-01


.. parsed-literal::

      72	 1.3287351e+00	 1.4006122e-01	 1.3812478e+00	 1.5331838e-01	[ 1.2343160e+00]	 2.2139311e-01


.. parsed-literal::

      73	 1.3346546e+00	 1.3960632e-01	 1.3869771e+00	 1.5319275e-01	[ 1.2427116e+00]	 2.0534039e-01


.. parsed-literal::

      74	 1.3417503e+00	 1.3891839e-01	 1.3941140e+00	 1.5273893e-01	[ 1.2526306e+00]	 2.2957587e-01


.. parsed-literal::

      75	 1.3473729e+00	 1.3838375e-01	 1.3997972e+00	 1.5244544e-01	[ 1.2608721e+00]	 2.2421074e-01


.. parsed-literal::

      76	 1.3546469e+00	 1.3760240e-01	 1.4072487e+00	 1.5164524e-01	[ 1.2727209e+00]	 2.2565746e-01


.. parsed-literal::

      77	 1.3620941e+00	 1.3723515e-01	 1.4147859e+00	 1.5108068e-01	[ 1.2806675e+00]	 2.1205544e-01


.. parsed-literal::

      78	 1.3658125e+00	 1.3714148e-01	 1.4185213e+00	 1.5079457e-01	[ 1.2832195e+00]	 2.2045255e-01


.. parsed-literal::

      79	 1.3721814e+00	 1.3682161e-01	 1.4250011e+00	 1.5031859e-01	[ 1.2879702e+00]	 2.1915269e-01


.. parsed-literal::

      80	 1.3775840e+00	 1.3629956e-01	 1.4308233e+00	 1.4979196e-01	  1.2836381e+00 	 2.2519231e-01


.. parsed-literal::

      81	 1.3834747e+00	 1.3603820e-01	 1.4365925e+00	 1.4947605e-01	[ 1.2925809e+00]	 2.2488856e-01


.. parsed-literal::

      82	 1.3880203e+00	 1.3574123e-01	 1.4410086e+00	 1.4955978e-01	[ 1.2974281e+00]	 2.2667694e-01


.. parsed-literal::

      83	 1.3931875e+00	 1.3536988e-01	 1.4463098e+00	 1.4947664e-01	[ 1.2992839e+00]	 2.0679379e-01


.. parsed-literal::

      84	 1.4002978e+00	 1.3462950e-01	 1.4536047e+00	 1.4968484e-01	[ 1.3005535e+00]	 2.2516775e-01


.. parsed-literal::

      85	 1.4068497e+00	 1.3416761e-01	 1.4602825e+00	 1.4958856e-01	[ 1.3019191e+00]	 2.3518181e-01


.. parsed-literal::

      86	 1.4108213e+00	 1.3406689e-01	 1.4642521e+00	 1.4944215e-01	[ 1.3062936e+00]	 2.3120999e-01


.. parsed-literal::

      87	 1.4162047e+00	 1.3387602e-01	 1.4698238e+00	 1.4932849e-01	[ 1.3088605e+00]	 2.1834755e-01


.. parsed-literal::

      88	 1.4201078e+00	 1.3327305e-01	 1.4736639e+00	 1.4898174e-01	[ 1.3142020e+00]	 2.3317695e-01


.. parsed-literal::

      89	 1.4249742e+00	 1.3290916e-01	 1.4785301e+00	 1.4881512e-01	[ 1.3148923e+00]	 2.3446083e-01


.. parsed-literal::

      90	 1.4300230e+00	 1.3225052e-01	 1.4837379e+00	 1.4844899e-01	  1.3117854e+00 	 2.1789742e-01


.. parsed-literal::

      91	 1.4333108e+00	 1.3192931e-01	 1.4871357e+00	 1.4818126e-01	  1.3093009e+00 	 2.3569441e-01


.. parsed-literal::

      92	 1.4386942e+00	 1.3102613e-01	 1.4927812e+00	 1.4789895e-01	  1.3032164e+00 	 2.1810412e-01


.. parsed-literal::

      93	 1.4436930e+00	 1.3070963e-01	 1.4977274e+00	 1.4738408e-01	  1.3022390e+00 	 2.2651291e-01


.. parsed-literal::

      94	 1.4463737e+00	 1.3070439e-01	 1.5002375e+00	 1.4716221e-01	  1.3118451e+00 	 2.1624398e-01


.. parsed-literal::

      95	 1.4498440e+00	 1.3034420e-01	 1.5036922e+00	 1.4690385e-01	[ 1.3164540e+00]	 2.1939588e-01


.. parsed-literal::

      96	 1.4541568e+00	 1.2963067e-01	 1.5081300e+00	 1.4617395e-01	[ 1.3215573e+00]	 2.2052169e-01


.. parsed-literal::

      97	 1.4577940e+00	 1.2884439e-01	 1.5119508e+00	 1.4554843e-01	  1.3187741e+00 	 2.3029041e-01


.. parsed-literal::

      98	 1.4613872e+00	 1.2860124e-01	 1.5154204e+00	 1.4537815e-01	  1.3194511e+00 	 2.2517085e-01


.. parsed-literal::

      99	 1.4641369e+00	 1.2852863e-01	 1.5181073e+00	 1.4539132e-01	  1.3191609e+00 	 2.1973991e-01


.. parsed-literal::

     100	 1.4667942e+00	 1.2814911e-01	 1.5209382e+00	 1.4539192e-01	  1.3151934e+00 	 2.0407200e-01
     101	 1.4698871e+00	 1.2797176e-01	 1.5241047e+00	 1.4527897e-01	  1.3137975e+00 	 2.0050073e-01


.. parsed-literal::

     102	 1.4723944e+00	 1.2781940e-01	 1.5267035e+00	 1.4507936e-01	  1.3146963e+00 	 2.2602129e-01


.. parsed-literal::

     103	 1.4758319e+00	 1.2735989e-01	 1.5302445e+00	 1.4454187e-01	  1.3176205e+00 	 2.3676229e-01


.. parsed-literal::

     104	 1.4787826e+00	 1.2664037e-01	 1.5334941e+00	 1.4385611e-01	  1.3148535e+00 	 2.2209811e-01


.. parsed-literal::

     105	 1.4815636e+00	 1.2633588e-01	 1.5361821e+00	 1.4355762e-01	  1.3196239e+00 	 2.1064663e-01


.. parsed-literal::

     106	 1.4841655e+00	 1.2589467e-01	 1.5387522e+00	 1.4332275e-01	[ 1.3221479e+00]	 2.2466660e-01


.. parsed-literal::

     107	 1.4866229e+00	 1.2555947e-01	 1.5412271e+00	 1.4325565e-01	[ 1.3226606e+00]	 2.1533060e-01


.. parsed-literal::

     108	 1.4885676e+00	 1.2516966e-01	 1.5432216e+00	 1.4326177e-01	[ 1.3261291e+00]	 3.7814975e-01


.. parsed-literal::

     109	 1.4909436e+00	 1.2492367e-01	 1.5456215e+00	 1.4340895e-01	  1.3232290e+00 	 2.1943831e-01


.. parsed-literal::

     110	 1.4929010e+00	 1.2482598e-01	 1.5476024e+00	 1.4348510e-01	  1.3223576e+00 	 2.2984219e-01


.. parsed-literal::

     111	 1.4960076e+00	 1.2452874e-01	 1.5507937e+00	 1.4368383e-01	  1.3204294e+00 	 2.1843815e-01


.. parsed-literal::

     112	 1.4987008e+00	 1.2416047e-01	 1.5535814e+00	 1.4340983e-01	  1.3195855e+00 	 2.1569419e-01


.. parsed-literal::

     113	 1.5007725e+00	 1.2396348e-01	 1.5556367e+00	 1.4331205e-01	  1.3195705e+00 	 2.2595477e-01


.. parsed-literal::

     114	 1.5035942e+00	 1.2356275e-01	 1.5584744e+00	 1.4310302e-01	  1.3176090e+00 	 2.2394490e-01
     115	 1.5050076e+00	 1.2339925e-01	 1.5599709e+00	 1.4306474e-01	  1.3133360e+00 	 1.9639254e-01


.. parsed-literal::

     116	 1.5067535e+00	 1.2331290e-01	 1.5617056e+00	 1.4304622e-01	  1.3133296e+00 	 2.2683454e-01


.. parsed-literal::

     117	 1.5092443e+00	 1.2311585e-01	 1.5643030e+00	 1.4324421e-01	  1.3081436e+00 	 2.1518993e-01


.. parsed-literal::

     118	 1.5108151e+00	 1.2296892e-01	 1.5659257e+00	 1.4329752e-01	  1.3076238e+00 	 2.2162724e-01


.. parsed-literal::

     119	 1.5133845e+00	 1.2274049e-01	 1.5686037e+00	 1.4347595e-01	  1.3047165e+00 	 2.2227716e-01


.. parsed-literal::

     120	 1.5142960e+00	 1.2242330e-01	 1.5697336e+00	 1.4369193e-01	  1.2951467e+00 	 2.3618603e-01


.. parsed-literal::

     121	 1.5168330e+00	 1.2239685e-01	 1.5721542e+00	 1.4339637e-01	  1.3021609e+00 	 2.2435427e-01


.. parsed-literal::

     122	 1.5177879e+00	 1.2229770e-01	 1.5731025e+00	 1.4326722e-01	  1.3020919e+00 	 2.2466207e-01


.. parsed-literal::

     123	 1.5198187e+00	 1.2207354e-01	 1.5751819e+00	 1.4308551e-01	  1.3000600e+00 	 2.0697165e-01


.. parsed-literal::

     124	 1.5223139e+00	 1.2185643e-01	 1.5777737e+00	 1.4309415e-01	  1.2945950e+00 	 2.2376609e-01


.. parsed-literal::

     125	 1.5232641e+00	 1.2161455e-01	 1.5789869e+00	 1.4315644e-01	  1.2876512e+00 	 2.1457434e-01


.. parsed-literal::

     126	 1.5257135e+00	 1.2162136e-01	 1.5813092e+00	 1.4321324e-01	  1.2894661e+00 	 2.2356462e-01


.. parsed-literal::

     127	 1.5266330e+00	 1.2165124e-01	 1.5821823e+00	 1.4321742e-01	  1.2915643e+00 	 2.2359252e-01


.. parsed-literal::

     128	 1.5286031e+00	 1.2155487e-01	 1.5841824e+00	 1.4326115e-01	  1.2918625e+00 	 2.3259830e-01


.. parsed-literal::

     129	 1.5305906e+00	 1.2145131e-01	 1.5862934e+00	 1.4304207e-01	  1.2891051e+00 	 2.1835470e-01


.. parsed-literal::

     130	 1.5328290e+00	 1.2119547e-01	 1.5885280e+00	 1.4306537e-01	  1.2904999e+00 	 2.2579694e-01


.. parsed-literal::

     131	 1.5343276e+00	 1.2105291e-01	 1.5899932e+00	 1.4297047e-01	  1.2907123e+00 	 2.2885036e-01


.. parsed-literal::

     132	 1.5360469e+00	 1.2090081e-01	 1.5917322e+00	 1.4305485e-01	  1.2861316e+00 	 2.2538018e-01


.. parsed-literal::

     133	 1.5375663e+00	 1.2076063e-01	 1.5932976e+00	 1.4288546e-01	  1.2894778e+00 	 2.3357129e-01


.. parsed-literal::

     134	 1.5388617e+00	 1.2077259e-01	 1.5945637e+00	 1.4297428e-01	  1.2896223e+00 	 2.3131990e-01


.. parsed-literal::

     135	 1.5408252e+00	 1.2079530e-01	 1.5965785e+00	 1.4329385e-01	  1.2876315e+00 	 2.2167754e-01


.. parsed-literal::

     136	 1.5420525e+00	 1.2078240e-01	 1.5978479e+00	 1.4335104e-01	  1.2867870e+00 	 2.3767662e-01


.. parsed-literal::

     137	 1.5439390e+00	 1.2067503e-01	 1.5997780e+00	 1.4343529e-01	  1.2840219e+00 	 2.2531581e-01


.. parsed-literal::

     138	 1.5458170e+00	 1.2033169e-01	 1.6018488e+00	 1.4345216e-01	  1.2757788e+00 	 2.2580099e-01


.. parsed-literal::

     139	 1.5472176e+00	 1.2017486e-01	 1.6032797e+00	 1.4319868e-01	  1.2766093e+00 	 2.2005558e-01


.. parsed-literal::

     140	 1.5483860e+00	 1.2014389e-01	 1.6043441e+00	 1.4315546e-01	  1.2786862e+00 	 2.1484733e-01


.. parsed-literal::

     141	 1.5496334e+00	 1.1999626e-01	 1.6055850e+00	 1.4312724e-01	  1.2786740e+00 	 2.0636225e-01


.. parsed-literal::

     142	 1.5514099e+00	 1.1968647e-01	 1.6074258e+00	 1.4304170e-01	  1.2776917e+00 	 2.2286868e-01


.. parsed-literal::

     143	 1.5529707e+00	 1.1936399e-01	 1.6090961e+00	 1.4299973e-01	  1.2720063e+00 	 2.2887087e-01


.. parsed-literal::

     144	 1.5542648e+00	 1.1902479e-01	 1.6104785e+00	 1.4278188e-01	  1.2741438e+00 	 2.0951295e-01


.. parsed-literal::

     145	 1.5552374e+00	 1.1899877e-01	 1.6114131e+00	 1.4281233e-01	  1.2721415e+00 	 2.1424103e-01


.. parsed-literal::

     146	 1.5559877e+00	 1.1895402e-01	 1.6121407e+00	 1.4276724e-01	  1.2706383e+00 	 2.2139978e-01


.. parsed-literal::

     147	 1.5574153e+00	 1.1878670e-01	 1.6135800e+00	 1.4263884e-01	  1.2668580e+00 	 2.1272111e-01


.. parsed-literal::

     148	 1.5587341e+00	 1.1857864e-01	 1.6149569e+00	 1.4226450e-01	  1.2620086e+00 	 2.2361541e-01


.. parsed-literal::

     149	 1.5600586e+00	 1.1844396e-01	 1.6162828e+00	 1.4214661e-01	  1.2611989e+00 	 2.1941447e-01


.. parsed-literal::

     150	 1.5611267e+00	 1.1833875e-01	 1.6173546e+00	 1.4197408e-01	  1.2621140e+00 	 2.2882795e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 17s, sys: 1.17 s, total: 2min 18s
    Wall time: 34.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff25c54aed0>



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
    CPU times: user 1.78 s, sys: 67.9 ms, total: 1.84 s
    Wall time: 603 ms


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




.. image:: 06_GPz_files/06_GPz_16_1.png


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




.. image:: 06_GPz_files/06_GPz_19_1.png

