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
       1	-3.5218519e-01	 3.2318788e-01	-3.4247097e-01	 3.1121379e-01	[-3.1824840e-01]	 4.6576500e-01


.. parsed-literal::

       2	-2.8217771e-01	 3.1314885e-01	-2.5923878e-01	 3.0441091e-01	[-2.2429143e-01]	 2.3117304e-01


.. parsed-literal::

       3	-2.3527105e-01	 2.9078371e-01	-1.9289850e-01	 2.8063799e-01	[-1.4385629e-01]	 2.9240155e-01
       4	-2.0601609e-01	 2.6752426e-01	-1.6637369e-01	 2.6003990e-01	[-1.1387439e-01]	 1.7521596e-01


.. parsed-literal::

       5	-1.0934722e-01	 2.5920249e-01	-7.5304176e-02	 2.4977245e-01	[-2.5731286e-02]	 2.1021032e-01


.. parsed-literal::

       6	-7.8174874e-02	 2.5373549e-01	-4.7894380e-02	 2.4399499e-01	[-7.9877833e-03]	 2.0575619e-01
       7	-5.8654191e-02	 2.5075804e-01	-3.6083831e-02	 2.3948440e-01	[ 8.8983252e-03]	 1.9069242e-01


.. parsed-literal::

       8	-4.7337675e-02	 2.4876464e-01	-2.7852216e-02	 2.3780256e-01	[ 1.5846595e-02]	 2.1707010e-01


.. parsed-literal::

       9	-3.3224451e-02	 2.4608410e-01	-1.6438917e-02	 2.3559405e-01	[ 2.6578722e-02]	 2.0999217e-01


.. parsed-literal::

      10	-2.6999656e-02	 2.4497455e-01	-1.1990626e-02	 2.3488852e-01	  2.6203609e-02 	 2.1642089e-01


.. parsed-literal::

      11	-2.0089032e-02	 2.4368568e-01	-5.6281804e-03	 2.3357824e-01	[ 3.6038665e-02]	 2.1921873e-01
      12	-1.7331885e-02	 2.4309324e-01	-2.9408299e-03	 2.3277864e-01	[ 3.9201277e-02]	 1.7118192e-01


.. parsed-literal::

      13	-1.2309860e-02	 2.4200458e-01	 2.2966635e-03	 2.3171096e-01	[ 4.5907484e-02]	 2.0929694e-01


.. parsed-literal::

      14	 1.4087986e-03	 2.3897965e-01	 1.7771610e-02	 2.2961143e-01	[ 6.0735958e-02]	 2.1102929e-01


.. parsed-literal::

      15	 4.4357777e-02	 2.2257446e-01	 6.3377641e-02	 2.2894667e-01	[ 7.8230398e-02]	 2.0541668e-01


.. parsed-literal::

      16	 1.5862809e-01	 2.1862800e-01	 1.8081207e-01	 2.2722105e-01	[ 2.0092931e-01]	 2.0364380e-01


.. parsed-literal::

      17	 1.9767139e-01	 2.1458406e-01	 2.2763212e-01	 2.0792576e-01	[ 2.8484155e-01]	 2.1209192e-01


.. parsed-literal::

      18	 2.8273531e-01	 2.1417266e-01	 3.1276081e-01	 2.0897565e-01	[ 3.4846473e-01]	 2.1742249e-01


.. parsed-literal::

      19	 3.1757658e-01	 2.1019347e-01	 3.4962686e-01	 2.1419816e-01	[ 3.7246221e-01]	 2.0424438e-01


.. parsed-literal::

      20	 3.5988028e-01	 2.0771811e-01	 3.9120254e-01	 2.1212446e-01	[ 4.1224722e-01]	 2.1429896e-01


.. parsed-literal::

      21	 4.1171215e-01	 2.0722003e-01	 4.4360456e-01	 2.0680931e-01	[ 4.7084182e-01]	 2.1837926e-01


.. parsed-literal::

      22	 4.8059791e-01	 2.0580963e-01	 5.1572771e-01	 1.9857186e-01	[ 5.4530472e-01]	 2.0893264e-01
      23	 5.2247173e-01	 2.0171037e-01	 5.6087901e-01	 1.9967231e-01	[ 5.9416940e-01]	 1.7422843e-01


.. parsed-literal::

      24	 5.6208075e-01	 1.9908460e-01	 6.0085691e-01	 1.9334873e-01	[ 6.3014208e-01]	 1.9842768e-01


.. parsed-literal::

      25	 6.0020068e-01	 1.9696767e-01	 6.3829554e-01	 1.9037235e-01	[ 6.6160683e-01]	 2.0934606e-01


.. parsed-literal::

      26	 6.4741860e-01	 1.9548991e-01	 6.8400003e-01	 1.8752307e-01	[ 6.9699057e-01]	 2.1344709e-01


.. parsed-literal::

      27	 6.7988490e-01	 2.0156764e-01	 7.1561633e-01	 1.9355105e-01	[ 7.2659908e-01]	 2.1628475e-01


.. parsed-literal::

      28	 7.0671760e-01	 2.0166482e-01	 7.4317707e-01	 1.9577017e-01	[ 7.4753219e-01]	 2.3237991e-01


.. parsed-literal::

      29	 7.2461708e-01	 2.0113844e-01	 7.6205238e-01	 1.9633243e-01	[ 7.6108333e-01]	 2.2038579e-01


.. parsed-literal::

      30	 7.4677922e-01	 2.0116848e-01	 7.8442660e-01	 1.9617655e-01	[ 7.8313539e-01]	 2.1802592e-01


.. parsed-literal::

      31	 7.6362461e-01	 2.0509858e-01	 8.0096946e-01	 2.0017025e-01	[ 8.0034462e-01]	 3.1446934e-01


.. parsed-literal::

      32	 7.8887485e-01	 2.1096918e-01	 8.2857157e-01	 2.0270614e-01	[ 8.3135410e-01]	 2.1954942e-01


.. parsed-literal::

      33	 8.1401951e-01	 2.1079864e-01	 8.5267663e-01	 2.0279843e-01	[ 8.5033072e-01]	 2.0906758e-01


.. parsed-literal::

      34	 8.3052339e-01	 2.0726960e-01	 8.6907441e-01	 2.0415068e-01	[ 8.6403351e-01]	 2.1116781e-01


.. parsed-literal::

      35	 8.5198418e-01	 2.0420101e-01	 8.9132711e-01	 2.1697705e-01	[ 8.7588001e-01]	 2.1340799e-01


.. parsed-literal::

      36	 8.6982962e-01	 2.0311449e-01	 9.0950022e-01	 2.1031329e-01	[ 8.9646078e-01]	 2.0916033e-01


.. parsed-literal::

      37	 8.8835195e-01	 2.0120211e-01	 9.2896879e-01	 2.0282615e-01	[ 9.1626640e-01]	 2.1980214e-01


.. parsed-literal::

      38	 9.0285254e-01	 2.0128638e-01	 9.4368409e-01	 1.9932028e-01	[ 9.3292231e-01]	 2.0217657e-01


.. parsed-literal::

      39	 9.2153044e-01	 1.9959218e-01	 9.6329217e-01	 1.9267776e-01	[ 9.5179002e-01]	 2.1976328e-01


.. parsed-literal::

      40	 9.4380247e-01	 1.9854508e-01	 9.8652730e-01	 1.8961332e-01	[ 9.6710038e-01]	 2.0850778e-01


.. parsed-literal::

      41	 9.6585347e-01	 1.9674781e-01	 1.0103564e+00	 1.8833433e-01	[ 9.8862371e-01]	 2.1896148e-01


.. parsed-literal::

      42	 9.7616297e-01	 1.9698895e-01	 1.0212947e+00	 1.8902264e-01	[ 9.9501882e-01]	 2.1043134e-01


.. parsed-literal::

      43	 9.8632772e-01	 1.9653862e-01	 1.0310283e+00	 1.8827967e-01	[ 1.0063037e+00]	 2.1898437e-01
      44	 9.9676581e-01	 1.9603259e-01	 1.0414397e+00	 1.8799029e-01	[ 1.0194715e+00]	 1.8641996e-01


.. parsed-literal::

      45	 1.0081240e+00	 1.9514605e-01	 1.0529133e+00	 1.8829120e-01	[ 1.0311442e+00]	 1.7663503e-01


.. parsed-literal::

      46	 1.0199246e+00	 1.9357541e-01	 1.0652703e+00	 1.8845099e-01	[ 1.0376986e+00]	 2.1003509e-01


.. parsed-literal::

      47	 1.0290366e+00	 1.9186792e-01	 1.0746087e+00	 1.8951620e-01	[ 1.0426402e+00]	 2.1739578e-01


.. parsed-literal::

      48	 1.0387822e+00	 1.8929166e-01	 1.0848116e+00	 1.9099487e-01	[ 1.0484384e+00]	 2.0859146e-01
      49	 1.0485431e+00	 1.8611342e-01	 1.0950901e+00	 1.8971690e-01	[ 1.0530535e+00]	 1.8650079e-01


.. parsed-literal::

      50	 1.0589325e+00	 1.8259962e-01	 1.1057177e+00	 1.8744343e-01	[ 1.0661675e+00]	 2.1924210e-01


.. parsed-literal::

      51	 1.0671129e+00	 1.8087344e-01	 1.1138812e+00	 1.8201912e-01	[ 1.0759838e+00]	 2.1737218e-01
      52	 1.0777950e+00	 1.7801386e-01	 1.1247524e+00	 1.7511796e-01	[ 1.0886267e+00]	 1.9379544e-01


.. parsed-literal::

      53	 1.0885093e+00	 1.7609661e-01	 1.1353564e+00	 1.7147387e-01	[ 1.1013546e+00]	 2.1636510e-01


.. parsed-literal::

      54	 1.0972697e+00	 1.7424452e-01	 1.1440498e+00	 1.7097359e-01	[ 1.1058467e+00]	 2.1504283e-01


.. parsed-literal::

      55	 1.1044686e+00	 1.7146102e-01	 1.1514951e+00	 1.7099155e-01	[ 1.1092088e+00]	 2.2054791e-01
      56	 1.1107750e+00	 1.7025163e-01	 1.1578380e+00	 1.7140900e-01	[ 1.1121333e+00]	 1.9824934e-01


.. parsed-literal::

      57	 1.1189953e+00	 1.6918781e-01	 1.1662630e+00	 1.7039251e-01	[ 1.1193778e+00]	 2.1315765e-01
      58	 1.1317444e+00	 1.6649011e-01	 1.1796885e+00	 1.6526760e-01	[ 1.1356634e+00]	 1.9891191e-01


.. parsed-literal::

      59	 1.1401474e+00	 1.6637513e-01	 1.1888081e+00	 1.6205926e-01	[ 1.1417475e+00]	 2.0485449e-01
      60	 1.1501275e+00	 1.6439921e-01	 1.1986959e+00	 1.5987864e-01	[ 1.1531885e+00]	 1.9942164e-01


.. parsed-literal::

      61	 1.1568685e+00	 1.6335057e-01	 1.2052528e+00	 1.5863286e-01	[ 1.1589525e+00]	 2.1622276e-01
      62	 1.1662901e+00	 1.6116162e-01	 1.2150340e+00	 1.5568675e-01	[ 1.1647607e+00]	 1.9239330e-01


.. parsed-literal::

      63	 1.1755105e+00	 1.5999892e-01	 1.2246778e+00	 1.5311315e-01	[ 1.1691426e+00]	 2.1731186e-01
      64	 1.1833090e+00	 1.5836949e-01	 1.2325864e+00	 1.5069567e-01	[ 1.1754645e+00]	 1.8455982e-01


.. parsed-literal::

      65	 1.1911504e+00	 1.5675301e-01	 1.2404707e+00	 1.4861812e-01	[ 1.1825475e+00]	 2.1500802e-01


.. parsed-literal::

      66	 1.1986043e+00	 1.5566487e-01	 1.2482202e+00	 1.4765615e-01	[ 1.1917820e+00]	 2.0865345e-01
      67	 1.2064171e+00	 1.5317834e-01	 1.2563241e+00	 1.4522611e-01	[ 1.1948972e+00]	 1.8833375e-01


.. parsed-literal::

      68	 1.2123029e+00	 1.5257200e-01	 1.2620908e+00	 1.4504950e-01	[ 1.2004650e+00]	 2.0925426e-01
      69	 1.2215279e+00	 1.5068601e-01	 1.2715195e+00	 1.4493051e-01	[ 1.2050011e+00]	 1.8668175e-01


.. parsed-literal::

      70	 1.2290229e+00	 1.4975404e-01	 1.2795407e+00	 1.4427778e-01	[ 1.2089697e+00]	 1.8865728e-01


.. parsed-literal::

      71	 1.2360404e+00	 1.4926531e-01	 1.2865499e+00	 1.4367952e-01	[ 1.2129676e+00]	 2.0648146e-01
      72	 1.2471075e+00	 1.4786099e-01	 1.2979589e+00	 1.4153855e-01	[ 1.2141014e+00]	 1.9927263e-01


.. parsed-literal::

      73	 1.2539219e+00	 1.4779019e-01	 1.3048197e+00	 1.4136439e-01	[ 1.2199416e+00]	 2.1645284e-01


.. parsed-literal::

      74	 1.2632127e+00	 1.4693962e-01	 1.3141136e+00	 1.4051668e-01	[ 1.2271509e+00]	 2.0917869e-01


.. parsed-literal::

      75	 1.2702207e+00	 1.4576668e-01	 1.3213434e+00	 1.3979533e-01	[ 1.2319320e+00]	 2.0934534e-01


.. parsed-literal::

      76	 1.2783783e+00	 1.4450666e-01	 1.3295249e+00	 1.3872068e-01	[ 1.2401890e+00]	 2.0313859e-01


.. parsed-literal::

      77	 1.2831711e+00	 1.4195634e-01	 1.3348874e+00	 1.3598626e-01	[ 1.2437547e+00]	 2.0739412e-01


.. parsed-literal::

      78	 1.2905699e+00	 1.4151567e-01	 1.3419691e+00	 1.3562500e-01	[ 1.2493606e+00]	 2.0559669e-01
      79	 1.2948961e+00	 1.4137376e-01	 1.3462877e+00	 1.3554883e-01	[ 1.2533635e+00]	 1.8551993e-01


.. parsed-literal::

      80	 1.3022869e+00	 1.4055632e-01	 1.3540500e+00	 1.3505519e-01	[ 1.2558840e+00]	 2.0319533e-01
      81	 1.3081226e+00	 1.3924029e-01	 1.3602967e+00	 1.3426191e-01	[ 1.2651525e+00]	 1.7276382e-01


.. parsed-literal::

      82	 1.3150825e+00	 1.3902201e-01	 1.3672848e+00	 1.3471755e-01	[ 1.2698363e+00]	 2.0411539e-01


.. parsed-literal::

      83	 1.3188766e+00	 1.3832656e-01	 1.3712596e+00	 1.3450787e-01	[ 1.2702822e+00]	 2.0587325e-01


.. parsed-literal::

      84	 1.3232876e+00	 1.3764414e-01	 1.3758094e+00	 1.3420743e-01	[ 1.2730351e+00]	 2.0450497e-01


.. parsed-literal::

      85	 1.3294460e+00	 1.3592096e-01	 1.3822547e+00	 1.3329903e-01	  1.2696289e+00 	 2.0402932e-01
      86	 1.3359536e+00	 1.3552569e-01	 1.3886661e+00	 1.3284133e-01	[ 1.2780284e+00]	 1.7111278e-01


.. parsed-literal::

      87	 1.3406586e+00	 1.3518091e-01	 1.3932789e+00	 1.3246505e-01	[ 1.2834139e+00]	 2.0781994e-01


.. parsed-literal::

      88	 1.3484006e+00	 1.3428021e-01	 1.4010450e+00	 1.3130700e-01	[ 1.2922301e+00]	 2.0367861e-01


.. parsed-literal::

      89	 1.3505558e+00	 1.3347265e-01	 1.4034801e+00	 1.3077846e-01	[ 1.2967068e+00]	 2.1081614e-01


.. parsed-literal::

      90	 1.3591283e+00	 1.3300629e-01	 1.4118195e+00	 1.2988189e-01	[ 1.3052470e+00]	 2.1020818e-01
      91	 1.3625877e+00	 1.3271378e-01	 1.4152400e+00	 1.2929588e-01	[ 1.3083966e+00]	 2.0079708e-01


.. parsed-literal::

      92	 1.3673596e+00	 1.3205244e-01	 1.4200852e+00	 1.2855254e-01	[ 1.3109143e+00]	 1.7419839e-01


.. parsed-literal::

      93	 1.3727041e+00	 1.3168364e-01	 1.4255018e+00	 1.2760683e-01	[ 1.3146306e+00]	 2.0563626e-01


.. parsed-literal::

      94	 1.3771710e+00	 1.3107698e-01	 1.4300404e+00	 1.2806471e-01	  1.3133304e+00 	 2.0578599e-01


.. parsed-literal::

      95	 1.3820683e+00	 1.3109638e-01	 1.4348488e+00	 1.2787956e-01	[ 1.3172518e+00]	 2.2090626e-01


.. parsed-literal::

      96	 1.3869614e+00	 1.3104332e-01	 1.4397548e+00	 1.2788703e-01	[ 1.3191766e+00]	 2.1077275e-01
      97	 1.3921374e+00	 1.3068303e-01	 1.4450807e+00	 1.2756515e-01	[ 1.3203724e+00]	 2.0097160e-01


.. parsed-literal::

      98	 1.3997935e+00	 1.2991020e-01	 1.4529387e+00	 1.2736224e-01	[ 1.3209828e+00]	 1.7588258e-01


.. parsed-literal::

      99	 1.4036072e+00	 1.2930789e-01	 1.4569590e+00	 1.2650331e-01	[ 1.3231544e+00]	 3.0110836e-01


.. parsed-literal::

     100	 1.4076114e+00	 1.2881111e-01	 1.4609743e+00	 1.2607914e-01	[ 1.3257637e+00]	 2.0870399e-01


.. parsed-literal::

     101	 1.4128912e+00	 1.2785689e-01	 1.4664730e+00	 1.2589881e-01	[ 1.3275927e+00]	 2.1319032e-01


.. parsed-literal::

     102	 1.4163801e+00	 1.2756743e-01	 1.4699363e+00	 1.2502185e-01	[ 1.3346660e+00]	 2.0569754e-01


.. parsed-literal::

     103	 1.4197922e+00	 1.2750740e-01	 1.4733276e+00	 1.2506239e-01	[ 1.3374299e+00]	 2.1041584e-01


.. parsed-literal::

     104	 1.4264629e+00	 1.2746224e-01	 1.4801345e+00	 1.2477715e-01	[ 1.3430577e+00]	 2.0667076e-01


.. parsed-literal::

     105	 1.4291446e+00	 1.2776980e-01	 1.4828401e+00	 1.2512564e-01	[ 1.3453155e+00]	 2.0988917e-01
     106	 1.4322121e+00	 1.2760249e-01	 1.4858187e+00	 1.2482723e-01	[ 1.3499951e+00]	 1.7371035e-01


.. parsed-literal::

     107	 1.4362365e+00	 1.2739874e-01	 1.4898820e+00	 1.2449407e-01	[ 1.3549305e+00]	 2.0994258e-01


.. parsed-literal::

     108	 1.4387465e+00	 1.2730579e-01	 1.4924583e+00	 1.2424974e-01	[ 1.3562268e+00]	 2.1220899e-01


.. parsed-literal::

     109	 1.4415457e+00	 1.2744526e-01	 1.4957033e+00	 1.2458347e-01	  1.3491268e+00 	 2.1755958e-01


.. parsed-literal::

     110	 1.4469228e+00	 1.2722864e-01	 1.5009314e+00	 1.2379544e-01	  1.3529434e+00 	 2.1008945e-01


.. parsed-literal::

     111	 1.4487434e+00	 1.2715411e-01	 1.5027136e+00	 1.2370026e-01	  1.3532107e+00 	 2.0672178e-01


.. parsed-literal::

     112	 1.4524503e+00	 1.2701671e-01	 1.5064469e+00	 1.2339986e-01	  1.3534449e+00 	 2.0929050e-01
     113	 1.4564248e+00	 1.2667632e-01	 1.5105128e+00	 1.2424936e-01	  1.3539718e+00 	 1.7450047e-01


.. parsed-literal::

     114	 1.4600080e+00	 1.2652625e-01	 1.5142081e+00	 1.2383998e-01	[ 1.3595228e+00]	 1.8875408e-01


.. parsed-literal::

     115	 1.4625499e+00	 1.2647306e-01	 1.5167597e+00	 1.2378992e-01	[ 1.3624850e+00]	 2.0741367e-01


.. parsed-literal::

     116	 1.4648978e+00	 1.2625036e-01	 1.5191536e+00	 1.2415801e-01	[ 1.3659492e+00]	 2.0834494e-01


.. parsed-literal::

     117	 1.4686019e+00	 1.2581356e-01	 1.5230199e+00	 1.2461170e-01	[ 1.3671878e+00]	 2.0917702e-01


.. parsed-literal::

     118	 1.4697318e+00	 1.2506806e-01	 1.5244370e+00	 1.2607984e-01	  1.3634341e+00 	 2.1164489e-01


.. parsed-literal::

     119	 1.4735620e+00	 1.2504340e-01	 1.5280568e+00	 1.2556143e-01	[ 1.3674052e+00]	 2.1712351e-01


.. parsed-literal::

     120	 1.4749978e+00	 1.2493578e-01	 1.5294346e+00	 1.2545830e-01	[ 1.3683224e+00]	 2.0239472e-01


.. parsed-literal::

     121	 1.4779604e+00	 1.2458546e-01	 1.5323597e+00	 1.2567887e-01	[ 1.3695573e+00]	 2.1064878e-01


.. parsed-literal::

     122	 1.4815334e+00	 1.2408723e-01	 1.5359719e+00	 1.2625277e-01	[ 1.3714331e+00]	 2.0320845e-01


.. parsed-literal::

     123	 1.4839063e+00	 1.2387279e-01	 1.5384098e+00	 1.2656138e-01	[ 1.3719039e+00]	 3.2041693e-01
     124	 1.4864767e+00	 1.2364670e-01	 1.5410179e+00	 1.2701736e-01	[ 1.3746204e+00]	 1.9832897e-01


.. parsed-literal::

     125	 1.4889383e+00	 1.2352291e-01	 1.5435419e+00	 1.2705177e-01	[ 1.3783312e+00]	 2.0528817e-01


.. parsed-literal::

     126	 1.4909336e+00	 1.2341197e-01	 1.5456561e+00	 1.2723643e-01	[ 1.3798630e+00]	 2.1304178e-01
     127	 1.4934155e+00	 1.2334454e-01	 1.5481046e+00	 1.2666844e-01	[ 1.3842601e+00]	 2.0010042e-01


.. parsed-literal::

     128	 1.4955197e+00	 1.2327127e-01	 1.5501682e+00	 1.2612880e-01	[ 1.3869844e+00]	 2.0888638e-01


.. parsed-literal::

     129	 1.4974999e+00	 1.2315630e-01	 1.5521603e+00	 1.2604130e-01	[ 1.3884458e+00]	 2.1020198e-01


.. parsed-literal::

     130	 1.4989690e+00	 1.2292238e-01	 1.5538373e+00	 1.2617192e-01	  1.3861438e+00 	 2.0956540e-01
     131	 1.5020491e+00	 1.2279374e-01	 1.5568806e+00	 1.2666794e-01	[ 1.3897196e+00]	 1.8962312e-01


.. parsed-literal::

     132	 1.5034568e+00	 1.2269991e-01	 1.5583382e+00	 1.2719006e-01	[ 1.3902591e+00]	 2.0924926e-01


.. parsed-literal::

     133	 1.5054319e+00	 1.2254873e-01	 1.5604224e+00	 1.2788748e-01	  1.3901614e+00 	 2.0504260e-01


.. parsed-literal::

     134	 1.5081742e+00	 1.2238430e-01	 1.5632400e+00	 1.2852786e-01	  1.3901508e+00 	 2.1083903e-01


.. parsed-literal::

     135	 1.5101694e+00	 1.2215295e-01	 1.5653970e+00	 1.2909145e-01	[ 1.3911085e+00]	 2.0646930e-01
     136	 1.5122973e+00	 1.2209787e-01	 1.5673898e+00	 1.2858022e-01	[ 1.3946844e+00]	 2.0914030e-01


.. parsed-literal::

     137	 1.5135157e+00	 1.2208384e-01	 1.5685441e+00	 1.2811598e-01	[ 1.3975007e+00]	 2.0818353e-01


.. parsed-literal::

     138	 1.5154317e+00	 1.2190780e-01	 1.5704608e+00	 1.2772172e-01	[ 1.4009347e+00]	 2.1278858e-01


.. parsed-literal::

     139	 1.5168306e+00	 1.2173274e-01	 1.5720509e+00	 1.2766294e-01	[ 1.4024138e+00]	 2.0720553e-01


.. parsed-literal::

     140	 1.5191904e+00	 1.2155949e-01	 1.5743886e+00	 1.2793124e-01	[ 1.4033288e+00]	 2.1114469e-01


.. parsed-literal::

     141	 1.5206270e+00	 1.2137568e-01	 1.5759108e+00	 1.2854163e-01	  1.4020228e+00 	 2.0350647e-01


.. parsed-literal::

     142	 1.5220559e+00	 1.2121735e-01	 1.5774119e+00	 1.2905379e-01	  1.4013614e+00 	 2.0731997e-01


.. parsed-literal::

     143	 1.5246520e+00	 1.2097762e-01	 1.5801096e+00	 1.2950106e-01	  1.4017187e+00 	 2.1306753e-01


.. parsed-literal::

     144	 1.5258401e+00	 1.2084326e-01	 1.5813123e+00	 1.2996650e-01	  1.4030343e+00 	 3.3764935e-01
     145	 1.5274325e+00	 1.2077998e-01	 1.5829008e+00	 1.2977325e-01	[ 1.4046140e+00]	 1.7706680e-01


.. parsed-literal::

     146	 1.5290045e+00	 1.2074360e-01	 1.5844445e+00	 1.2920071e-01	[ 1.4061672e+00]	 1.9502950e-01
     147	 1.5304005e+00	 1.2069072e-01	 1.5859118e+00	 1.2898820e-01	  1.4049477e+00 	 1.9735074e-01


.. parsed-literal::

     148	 1.5319944e+00	 1.2066704e-01	 1.5875322e+00	 1.2864475e-01	  1.4051049e+00 	 2.0522094e-01


.. parsed-literal::

     149	 1.5336099e+00	 1.2060558e-01	 1.5892255e+00	 1.2888497e-01	  1.4035791e+00 	 2.1889329e-01


.. parsed-literal::

     150	 1.5360151e+00	 1.2052447e-01	 1.5917935e+00	 1.2980914e-01	  1.4006179e+00 	 2.1411514e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.15 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f22142d1c30>



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
    CPU times: user 2.09 s, sys: 55 ms, total: 2.15 s
    Wall time: 650 ms


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

