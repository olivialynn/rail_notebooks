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
       1	-3.5409516e-01	 3.2439942e-01	-3.4443708e-01	 3.0817391e-01	[-3.1100501e-01]	 4.6996379e-01


.. parsed-literal::

       2	-2.8976134e-01	 3.1631896e-01	-2.6837756e-01	 2.9558678e-01	[-2.0602787e-01]	 2.4175835e-01


.. parsed-literal::

       3	-2.4027127e-01	 2.9298855e-01	-1.9589639e-01	 2.7968868e-01	[-1.3103413e-01]	 2.9193616e-01


.. parsed-literal::

       4	-2.1335946e-01	 2.7891518e-01	-1.6627907e-01	 2.7435816e-01	[-1.1154676e-01]	 2.9897928e-01


.. parsed-literal::

       5	-1.4782499e-01	 2.6338695e-01	-1.0579476e-01	 2.5309843e-01	[-3.7671286e-02]	 2.1622038e-01


.. parsed-literal::

       6	-8.8472667e-02	 2.5654476e-01	-5.9565803e-02	 2.4367688e-01	[-1.9247595e-03]	 2.0673704e-01


.. parsed-literal::

       7	-7.1057368e-02	 2.5381446e-01	-4.7417536e-02	 2.3912836e-01	[ 1.2666973e-02]	 2.2422671e-01
       8	-5.4941791e-02	 2.5084488e-01	-3.6012924e-02	 2.3524624e-01	[ 2.7702939e-02]	 2.0024991e-01


.. parsed-literal::

       9	-4.4485127e-02	 2.4885387e-01	-2.7949158e-02	 2.3226671e-01	[ 3.3651520e-02]	 2.0697379e-01


.. parsed-literal::

      10	-3.3676886e-02	 2.4673157e-01	-1.8556114e-02	 2.2906485e-01	[ 5.2057940e-02]	 2.0584822e-01
      11	-2.4178536e-02	 2.4495397e-01	-9.8857667e-03	 2.2626519e-01	[ 6.1775448e-02]	 1.9421363e-01


.. parsed-literal::

      12	-1.8963826e-02	 2.4403006e-01	-4.8944119e-03	 2.2705831e-01	[ 6.2571811e-02]	 1.8630075e-01


.. parsed-literal::

      13	-1.3332712e-02	 2.4279653e-01	 1.4666611e-03	 2.2673479e-01	[ 6.6115710e-02]	 2.0750999e-01


.. parsed-literal::

      14	 7.9758337e-02	 2.3115634e-01	 9.8430314e-02	 2.1724118e-01	[ 1.5469903e-01]	 2.0362997e-01


.. parsed-literal::

      15	 1.3659303e-01	 2.2908692e-01	 1.6015212e-01	 2.1481308e-01	[ 1.9737031e-01]	 3.2918310e-01


.. parsed-literal::

      16	 1.8411232e-01	 2.2584691e-01	 2.0759418e-01	 2.1072514e-01	[ 2.4239145e-01]	 2.1560216e-01


.. parsed-literal::

      17	 2.5432253e-01	 2.2251805e-01	 2.8178956e-01	 2.0906439e-01	[ 3.1374473e-01]	 2.1376729e-01


.. parsed-literal::

      18	 3.2163224e-01	 2.1798627e-01	 3.5328882e-01	 2.1197688e-01	[ 3.7959663e-01]	 2.1327639e-01
      19	 3.9363161e-01	 2.1397907e-01	 4.2716515e-01	 2.0854560e-01	[ 4.5021061e-01]	 1.7400861e-01


.. parsed-literal::

      20	 4.6233471e-01	 2.1038551e-01	 4.9732356e-01	 2.0082172e-01	[ 5.1758640e-01]	 2.1401238e-01


.. parsed-literal::

      21	 5.2653456e-01	 2.0566019e-01	 5.6333998e-01	 1.9360199e-01	[ 5.7987888e-01]	 2.1012831e-01


.. parsed-literal::

      22	 5.8842170e-01	 2.0159295e-01	 6.2821117e-01	 1.9315372e-01	[ 6.2422160e-01]	 2.1442986e-01
      23	 6.3135783e-01	 1.9725659e-01	 6.7114474e-01	 1.8606945e-01	[ 6.6432380e-01]	 1.7889309e-01


.. parsed-literal::

      24	 6.6776245e-01	 1.9533714e-01	 7.0694179e-01	 1.8539718e-01	[ 6.9859167e-01]	 1.8760300e-01


.. parsed-literal::

      25	 6.9337077e-01	 1.9413333e-01	 7.3178508e-01	 1.8297912e-01	[ 7.2135683e-01]	 2.1591425e-01
      26	 7.1783342e-01	 1.9378412e-01	 7.5599412e-01	 1.8364235e-01	[ 7.4571838e-01]	 1.8050480e-01


.. parsed-literal::

      27	 7.4961240e-01	 2.0352373e-01	 7.8854313e-01	 1.9285761e-01	[ 7.9419865e-01]	 1.8463707e-01
      28	 7.9344182e-01	 2.0043630e-01	 8.3323724e-01	 1.9108056e-01	[ 8.2721433e-01]	 2.0161819e-01


.. parsed-literal::

      29	 8.2308563e-01	 1.9901405e-01	 8.6316127e-01	 1.8659750e-01	[ 8.5462380e-01]	 2.1872807e-01
      30	 8.4895698e-01	 1.9875122e-01	 8.8971640e-01	 1.8492641e-01	[ 8.8231927e-01]	 2.0044661e-01


.. parsed-literal::

      31	 8.8548692e-01	 1.9456380e-01	 9.2781241e-01	 1.7765370e-01	[ 9.2782416e-01]	 1.9122672e-01


.. parsed-literal::

      32	 8.8794271e-01	 1.9188086e-01	 9.3195221e-01	 1.7616080e-01	[ 9.2801431e-01]	 2.1554613e-01
      33	 9.1010171e-01	 1.8950340e-01	 9.5343545e-01	 1.7284405e-01	[ 9.5317296e-01]	 1.7995453e-01


.. parsed-literal::

      34	 9.2012197e-01	 1.8771030e-01	 9.6357646e-01	 1.7054747e-01	[ 9.6460720e-01]	 1.9866180e-01
      35	 9.3256696e-01	 1.8533332e-01	 9.7656915e-01	 1.6777558e-01	[ 9.7837635e-01]	 1.8289018e-01


.. parsed-literal::

      36	 9.6162771e-01	 1.7977767e-01	 1.0070852e+00	 1.6228524e-01	[ 1.0079315e+00]	 2.0957422e-01


.. parsed-literal::

      37	 9.7680719e-01	 1.7758896e-01	 1.0228620e+00	 1.5990659e-01	[ 1.0246060e+00]	 3.2634997e-01


.. parsed-literal::

      38	 9.9415206e-01	 1.7440524e-01	 1.0412352e+00	 1.5668317e-01	[ 1.0442530e+00]	 2.0253658e-01


.. parsed-literal::

      39	 1.0062690e+00	 1.7228527e-01	 1.0535563e+00	 1.5587751e-01	[ 1.0567867e+00]	 2.1186614e-01


.. parsed-literal::

      40	 1.0230632e+00	 1.6916065e-01	 1.0705318e+00	 1.5452918e-01	[ 1.0730842e+00]	 2.1630549e-01
      41	 1.0378060e+00	 1.6803777e-01	 1.0850270e+00	 1.5366154e-01	[ 1.0879137e+00]	 1.7874837e-01


.. parsed-literal::

      42	 1.0514239e+00	 1.6554527e-01	 1.0985975e+00	 1.5006821e-01	[ 1.1003336e+00]	 2.0801806e-01


.. parsed-literal::

      43	 1.0641268e+00	 1.6404207e-01	 1.1110710e+00	 1.4563403e-01	[ 1.1145060e+00]	 2.1059608e-01


.. parsed-literal::

      44	 1.0762411e+00	 1.6117126e-01	 1.1236743e+00	 1.4251401e-01	[ 1.1215252e+00]	 2.1253538e-01


.. parsed-literal::

      45	 1.0899436e+00	 1.5914250e-01	 1.1375638e+00	 1.4032246e-01	[ 1.1318975e+00]	 2.2384501e-01


.. parsed-literal::

      46	 1.1057353e+00	 1.5623124e-01	 1.1535843e+00	 1.3836580e-01	[ 1.1423969e+00]	 2.1714306e-01
      47	 1.1199816e+00	 1.5419882e-01	 1.1681158e+00	 1.3815209e-01	[ 1.1562352e+00]	 1.9523883e-01


.. parsed-literal::

      48	 1.1336712e+00	 1.5362170e-01	 1.1818271e+00	 1.3774527e-01	[ 1.1724548e+00]	 2.0912981e-01
      49	 1.1474333e+00	 1.5334968e-01	 1.1954676e+00	 1.3777216e-01	[ 1.1876498e+00]	 1.8701530e-01


.. parsed-literal::

      50	 1.1579327e+00	 1.5247992e-01	 1.2064662e+00	 1.3588218e-01	[ 1.1989827e+00]	 2.1688676e-01


.. parsed-literal::

      51	 1.1689279e+00	 1.5253520e-01	 1.2174087e+00	 1.3576796e-01	[ 1.2051734e+00]	 2.0513606e-01


.. parsed-literal::

      52	 1.1761050e+00	 1.5199774e-01	 1.2246111e+00	 1.3522645e-01	[ 1.2097468e+00]	 2.1021795e-01
      53	 1.1877151e+00	 1.5130678e-01	 1.2364385e+00	 1.3384774e-01	[ 1.2177112e+00]	 1.8767500e-01


.. parsed-literal::

      54	 1.1985091e+00	 1.5090259e-01	 1.2476548e+00	 1.3167846e-01	[ 1.2281579e+00]	 2.0680237e-01


.. parsed-literal::

      55	 1.2097226e+00	 1.4975799e-01	 1.2588498e+00	 1.3025592e-01	[ 1.2386713e+00]	 2.1220326e-01


.. parsed-literal::

      56	 1.2180918e+00	 1.4892070e-01	 1.2673027e+00	 1.2920480e-01	[ 1.2483496e+00]	 2.0743728e-01


.. parsed-literal::

      57	 1.2280053e+00	 1.4795350e-01	 1.2775831e+00	 1.2814602e-01	[ 1.2592826e+00]	 2.0180845e-01


.. parsed-literal::

      58	 1.2406341e+00	 1.4711317e-01	 1.2904532e+00	 1.2659562e-01	[ 1.2719310e+00]	 2.1257281e-01


.. parsed-literal::

      59	 1.2524392e+00	 1.4552930e-01	 1.3023851e+00	 1.2590535e-01	[ 1.2799071e+00]	 2.1330523e-01


.. parsed-literal::

      60	 1.2624521e+00	 1.4446149e-01	 1.3125251e+00	 1.2564485e-01	[ 1.2880302e+00]	 2.1246624e-01


.. parsed-literal::

      61	 1.2726983e+00	 1.4276682e-01	 1.3229567e+00	 1.2451259e-01	[ 1.2907977e+00]	 2.1176219e-01


.. parsed-literal::

      62	 1.2849235e+00	 1.4102725e-01	 1.3352998e+00	 1.2371718e-01	[ 1.3024429e+00]	 2.0771575e-01


.. parsed-literal::

      63	 1.2965281e+00	 1.4010114e-01	 1.3471771e+00	 1.2246827e-01	[ 1.3112650e+00]	 2.0744300e-01


.. parsed-literal::

      64	 1.3055693e+00	 1.3951127e-01	 1.3562170e+00	 1.2138809e-01	[ 1.3210314e+00]	 2.1100545e-01


.. parsed-literal::

      65	 1.3132743e+00	 1.3940426e-01	 1.3638070e+00	 1.2119891e-01	[ 1.3264109e+00]	 2.1455288e-01


.. parsed-literal::

      66	 1.3218940e+00	 1.3838349e-01	 1.3726651e+00	 1.2095584e-01	[ 1.3291143e+00]	 2.1211672e-01


.. parsed-literal::

      67	 1.3297478e+00	 1.3777775e-01	 1.3807585e+00	 1.2018946e-01	[ 1.3328166e+00]	 2.1645021e-01


.. parsed-literal::

      68	 1.3366691e+00	 1.3700675e-01	 1.3877239e+00	 1.2001243e-01	[ 1.3362010e+00]	 2.1795964e-01


.. parsed-literal::

      69	 1.3445954e+00	 1.3573336e-01	 1.3958055e+00	 1.1870896e-01	[ 1.3406855e+00]	 2.1967673e-01
      70	 1.3527678e+00	 1.3585665e-01	 1.4043376e+00	 1.1891765e-01	[ 1.3474685e+00]	 2.0024204e-01


.. parsed-literal::

      71	 1.3601569e+00	 1.3470700e-01	 1.4118738e+00	 1.1686551e-01	[ 1.3537826e+00]	 1.9920921e-01


.. parsed-literal::

      72	 1.3660359e+00	 1.3419713e-01	 1.4179386e+00	 1.1608253e-01	[ 1.3611806e+00]	 2.0520258e-01


.. parsed-literal::

      73	 1.3723446e+00	 1.3388484e-01	 1.4243311e+00	 1.1510429e-01	[ 1.3703028e+00]	 2.0637226e-01


.. parsed-literal::

      74	 1.3782947e+00	 1.3331252e-01	 1.4304532e+00	 1.1458976e-01	[ 1.3713161e+00]	 2.2637105e-01


.. parsed-literal::

      75	 1.3843109e+00	 1.3304456e-01	 1.4364679e+00	 1.1434656e-01	[ 1.3770900e+00]	 2.0564508e-01


.. parsed-literal::

      76	 1.3899614e+00	 1.3270451e-01	 1.4420768e+00	 1.1434504e-01	[ 1.3822439e+00]	 2.0741224e-01
      77	 1.3947064e+00	 1.3242277e-01	 1.4470059e+00	 1.1456238e-01	[ 1.3886867e+00]	 1.7538834e-01


.. parsed-literal::

      78	 1.3992568e+00	 1.3222689e-01	 1.4515876e+00	 1.1448522e-01	[ 1.3942101e+00]	 2.0008445e-01


.. parsed-literal::

      79	 1.4031198e+00	 1.3195609e-01	 1.4554969e+00	 1.1414104e-01	[ 1.3982311e+00]	 2.2990584e-01


.. parsed-literal::

      80	 1.4087138e+00	 1.3158567e-01	 1.4612371e+00	 1.1397385e-01	[ 1.4021744e+00]	 2.0944238e-01


.. parsed-literal::

      81	 1.4099416e+00	 1.3165926e-01	 1.4626099e+00	 1.1440343e-01	  1.4021490e+00 	 2.1894050e-01


.. parsed-literal::

      82	 1.4160493e+00	 1.3151199e-01	 1.4686247e+00	 1.1455755e-01	[ 1.4073818e+00]	 2.1769142e-01


.. parsed-literal::

      83	 1.4183180e+00	 1.3158128e-01	 1.4708834e+00	 1.1468215e-01	[ 1.4097704e+00]	 2.0933056e-01


.. parsed-literal::

      84	 1.4212555e+00	 1.3164281e-01	 1.4738836e+00	 1.1471269e-01	[ 1.4130615e+00]	 2.1966529e-01


.. parsed-literal::

      85	 1.4257319e+00	 1.3155695e-01	 1.4784712e+00	 1.1405844e-01	[ 1.4183658e+00]	 2.1062875e-01
      86	 1.4271200e+00	 1.3142788e-01	 1.4803201e+00	 1.1385458e-01	  1.4149357e+00 	 1.9069409e-01


.. parsed-literal::

      87	 1.4325618e+00	 1.3109305e-01	 1.4855156e+00	 1.1256975e-01	[ 1.4216962e+00]	 2.1846533e-01
      88	 1.4347685e+00	 1.3085059e-01	 1.4876651e+00	 1.1191349e-01	[ 1.4230742e+00]	 1.8385649e-01


.. parsed-literal::

      89	 1.4382613e+00	 1.3050705e-01	 1.4912175e+00	 1.1072899e-01	[ 1.4250434e+00]	 1.9369769e-01


.. parsed-literal::

      90	 1.4424142e+00	 1.3012993e-01	 1.4954123e+00	 1.0935278e-01	[ 1.4260452e+00]	 2.2195911e-01


.. parsed-literal::

      91	 1.4465237e+00	 1.3020014e-01	 1.4995477e+00	 1.0869932e-01	[ 1.4337148e+00]	 2.1420622e-01


.. parsed-literal::

      92	 1.4494799e+00	 1.3005213e-01	 1.5024830e+00	 1.0829706e-01	[ 1.4370738e+00]	 2.1964073e-01


.. parsed-literal::

      93	 1.4522991e+00	 1.2980350e-01	 1.5053105e+00	 1.0839338e-01	[ 1.4387227e+00]	 2.1684051e-01


.. parsed-literal::

      94	 1.4551637e+00	 1.2961722e-01	 1.5082198e+00	 1.0836493e-01	[ 1.4404597e+00]	 2.0807266e-01
      95	 1.4588703e+00	 1.2920557e-01	 1.5121405e+00	 1.0850415e-01	  1.4354145e+00 	 2.0267725e-01


.. parsed-literal::

      96	 1.4616265e+00	 1.2906387e-01	 1.5148538e+00	 1.0865654e-01	  1.4395698e+00 	 2.1471238e-01


.. parsed-literal::

      97	 1.4633743e+00	 1.2895221e-01	 1.5165535e+00	 1.0841632e-01	[ 1.4412942e+00]	 2.1132827e-01


.. parsed-literal::

      98	 1.4655918e+00	 1.2884536e-01	 1.5187832e+00	 1.0853804e-01	[ 1.4437985e+00]	 2.0595789e-01


.. parsed-literal::

      99	 1.4685543e+00	 1.2852894e-01	 1.5218505e+00	 1.0837044e-01	  1.4433956e+00 	 2.0748091e-01


.. parsed-literal::

     100	 1.4714870e+00	 1.2833464e-01	 1.5248442e+00	 1.0803661e-01	[ 1.4442324e+00]	 2.0987868e-01
     101	 1.4742153e+00	 1.2810494e-01	 1.5277018e+00	 1.0810049e-01	  1.4433026e+00 	 1.8410850e-01


.. parsed-literal::

     102	 1.4764057e+00	 1.2792513e-01	 1.5299634e+00	 1.0756011e-01	  1.4422213e+00 	 2.1863151e-01


.. parsed-literal::

     103	 1.4788045e+00	 1.2776288e-01	 1.5323910e+00	 1.0739495e-01	  1.4414478e+00 	 2.2118688e-01
     104	 1.4821528e+00	 1.2753010e-01	 1.5358824e+00	 1.0688258e-01	  1.4416599e+00 	 1.9394159e-01


.. parsed-literal::

     105	 1.4852278e+00	 1.2742423e-01	 1.5389019e+00	 1.0674612e-01	  1.4393108e+00 	 1.7684579e-01
     106	 1.4868041e+00	 1.2739481e-01	 1.5404391e+00	 1.0639589e-01	  1.4403819e+00 	 1.9394493e-01


.. parsed-literal::

     107	 1.4891117e+00	 1.2744085e-01	 1.5427280e+00	 1.0609360e-01	  1.4421500e+00 	 1.7998481e-01
     108	 1.4921738e+00	 1.2756924e-01	 1.5458687e+00	 1.0570902e-01	  1.4419128e+00 	 1.9700432e-01


.. parsed-literal::

     109	 1.4953131e+00	 1.2750483e-01	 1.5492088e+00	 1.0587420e-01	  1.4360937e+00 	 2.1206927e-01


.. parsed-literal::

     110	 1.4977809e+00	 1.2758187e-01	 1.5516458e+00	 1.0571489e-01	  1.4401206e+00 	 2.0989919e-01


.. parsed-literal::

     111	 1.4993310e+00	 1.2748411e-01	 1.5531867e+00	 1.0559624e-01	  1.4415587e+00 	 2.0637846e-01


.. parsed-literal::

     112	 1.5018424e+00	 1.2716362e-01	 1.5557431e+00	 1.0568377e-01	[ 1.4445855e+00]	 2.0894217e-01


.. parsed-literal::

     113	 1.5037707e+00	 1.2695690e-01	 1.5578546e+00	 1.0532237e-01	[ 1.4480050e+00]	 2.1468449e-01


.. parsed-literal::

     114	 1.5061142e+00	 1.2671487e-01	 1.5601336e+00	 1.0553679e-01	[ 1.4522399e+00]	 2.1222115e-01


.. parsed-literal::

     115	 1.5078134e+00	 1.2654253e-01	 1.5618319e+00	 1.0567091e-01	[ 1.4543789e+00]	 2.1192551e-01


.. parsed-literal::

     116	 1.5097930e+00	 1.2628005e-01	 1.5638549e+00	 1.0582563e-01	[ 1.4562549e+00]	 2.1292400e-01


.. parsed-literal::

     117	 1.5116467e+00	 1.2592682e-01	 1.5658124e+00	 1.0611790e-01	  1.4521163e+00 	 2.0748830e-01
     118	 1.5140706e+00	 1.2572794e-01	 1.5682696e+00	 1.0607865e-01	  1.4543161e+00 	 2.0055056e-01


.. parsed-literal::

     119	 1.5153030e+00	 1.2574736e-01	 1.5694696e+00	 1.0600602e-01	  1.4533657e+00 	 2.1666908e-01


.. parsed-literal::

     120	 1.5169691e+00	 1.2566705e-01	 1.5711856e+00	 1.0602468e-01	  1.4510908e+00 	 2.1284866e-01


.. parsed-literal::

     121	 1.5188945e+00	 1.2544589e-01	 1.5733041e+00	 1.0645708e-01	  1.4420352e+00 	 2.1901250e-01


.. parsed-literal::

     122	 1.5210670e+00	 1.2541577e-01	 1.5754668e+00	 1.0622888e-01	  1.4434842e+00 	 2.1180725e-01
     123	 1.5227666e+00	 1.2537417e-01	 1.5771735e+00	 1.0618482e-01	  1.4442720e+00 	 2.0663190e-01


.. parsed-literal::

     124	 1.5241832e+00	 1.2529898e-01	 1.5785947e+00	 1.0613989e-01	  1.4443834e+00 	 2.0956182e-01


.. parsed-literal::

     125	 1.5264471e+00	 1.2506106e-01	 1.5809239e+00	 1.0603513e-01	  1.4434569e+00 	 2.0736098e-01


.. parsed-literal::

     126	 1.5285784e+00	 1.2480099e-01	 1.5831231e+00	 1.0580780e-01	  1.4419325e+00 	 2.1650672e-01


.. parsed-literal::

     127	 1.5300906e+00	 1.2467375e-01	 1.5846403e+00	 1.0554122e-01	  1.4436718e+00 	 2.2101474e-01


.. parsed-literal::

     128	 1.5319183e+00	 1.2433090e-01	 1.5866058e+00	 1.0521193e-01	  1.4435928e+00 	 2.0901346e-01


.. parsed-literal::

     129	 1.5328259e+00	 1.2418772e-01	 1.5876202e+00	 1.0493887e-01	  1.4453923e+00 	 2.1096563e-01


.. parsed-literal::

     130	 1.5338978e+00	 1.2415620e-01	 1.5886662e+00	 1.0486891e-01	  1.4466709e+00 	 2.1859002e-01


.. parsed-literal::

     131	 1.5353062e+00	 1.2406365e-01	 1.5901189e+00	 1.0467764e-01	  1.4471550e+00 	 2.0950389e-01


.. parsed-literal::

     132	 1.5364324e+00	 1.2401880e-01	 1.5912426e+00	 1.0451895e-01	  1.4471464e+00 	 2.1480179e-01


.. parsed-literal::

     133	 1.5386319e+00	 1.2382568e-01	 1.5934509e+00	 1.0419048e-01	  1.4454372e+00 	 2.1227264e-01
     134	 1.5396340e+00	 1.2364740e-01	 1.5944955e+00	 1.0414036e-01	  1.4393331e+00 	 1.9246936e-01


.. parsed-literal::

     135	 1.5410623e+00	 1.2367458e-01	 1.5958210e+00	 1.0404733e-01	  1.4424966e+00 	 2.1399903e-01


.. parsed-literal::

     136	 1.5419549e+00	 1.2366476e-01	 1.5966805e+00	 1.0396411e-01	  1.4429523e+00 	 2.0935965e-01


.. parsed-literal::

     137	 1.5432797e+00	 1.2366285e-01	 1.5980022e+00	 1.0375527e-01	  1.4422883e+00 	 2.2253728e-01


.. parsed-literal::

     138	 1.5452350e+00	 1.2365962e-01	 1.6000046e+00	 1.0339114e-01	  1.4414690e+00 	 2.1315432e-01


.. parsed-literal::

     139	 1.5464922e+00	 1.2364340e-01	 1.6013168e+00	 1.0324219e-01	  1.4382276e+00 	 3.3407617e-01


.. parsed-literal::

     140	 1.5477814e+00	 1.2356764e-01	 1.6026372e+00	 1.0324675e-01	  1.4382008e+00 	 2.1365952e-01


.. parsed-literal::

     141	 1.5487121e+00	 1.2348293e-01	 1.6035958e+00	 1.0331406e-01	  1.4390565e+00 	 2.1907401e-01


.. parsed-literal::

     142	 1.5493877e+00	 1.2332514e-01	 1.6042729e+00	 1.0365041e-01	  1.4398014e+00 	 2.1918297e-01


.. parsed-literal::

     143	 1.5502380e+00	 1.2328969e-01	 1.6051120e+00	 1.0362927e-01	  1.4405436e+00 	 2.1279788e-01
     144	 1.5510797e+00	 1.2324647e-01	 1.6059655e+00	 1.0362875e-01	  1.4404081e+00 	 1.9580317e-01


.. parsed-literal::

     145	 1.5515853e+00	 1.2321053e-01	 1.6064778e+00	 1.0358897e-01	  1.4401466e+00 	 1.9159460e-01


.. parsed-literal::

     146	 1.5534605e+00	 1.2307425e-01	 1.6083734e+00	 1.0316053e-01	  1.4392604e+00 	 2.0847774e-01


.. parsed-literal::

     147	 1.5542938e+00	 1.2293145e-01	 1.6093341e+00	 1.0285531e-01	  1.4324382e+00 	 2.2620153e-01


.. parsed-literal::

     148	 1.5558731e+00	 1.2289116e-01	 1.6108676e+00	 1.0261348e-01	  1.4365291e+00 	 2.1528888e-01


.. parsed-literal::

     149	 1.5566878e+00	 1.2287077e-01	 1.6116686e+00	 1.0250158e-01	  1.4380663e+00 	 2.1440673e-01
     150	 1.5577395e+00	 1.2282217e-01	 1.6127605e+00	 1.0236940e-01	  1.4394671e+00 	 1.8826103e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.18 s, total: 2min 7s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc0b8d40d00>



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
    CPU times: user 1.85 s, sys: 49 ms, total: 1.9 s
    Wall time: 613 ms


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

