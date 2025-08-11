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
       1	-3.3836352e-01	 3.1890549e-01	-3.2878850e-01	 3.2754803e-01	[-3.4566801e-01]	 4.6248984e-01


.. parsed-literal::

       2	-2.6895081e-01	 3.0904071e-01	-2.4568120e-01	 3.1649586e-01	[-2.6915358e-01]	 2.3123646e-01


.. parsed-literal::

       3	-2.2092000e-01	 2.8588859e-01	-1.7638820e-01	 2.9239627e-01	[-2.0561678e-01]	 2.9189086e-01


.. parsed-literal::

       4	-1.8982902e-01	 2.6298818e-01	-1.4874594e-01	 2.7353620e-01	 -2.1098799e-01 	 2.0190287e-01
       5	-9.1260090e-02	 2.5479474e-01	-5.6921254e-02	 2.6275186e-01	[-9.4689796e-02]	 2.0051980e-01


.. parsed-literal::

       6	-6.3239542e-02	 2.5018375e-01	-3.3162170e-02	 2.5988477e-01	[-6.8618386e-02]	 2.0461059e-01


.. parsed-literal::

       7	-4.3615999e-02	 2.4692700e-01	-2.0525443e-02	 2.5544478e-01	[-5.4440338e-02]	 2.1254849e-01
       8	-3.2461731e-02	 2.4504481e-01	-1.2675415e-02	 2.5362496e-01	[-4.7467010e-02]	 1.9564891e-01


.. parsed-literal::

       9	-1.8161163e-02	 2.4230017e-01	-1.0231783e-03	 2.5138605e-01	[-3.6925964e-02]	 2.1106625e-01
      10	-7.1465067e-03	 2.4002940e-01	 8.1904236e-03	 2.4998755e-01	[-3.2449513e-02]	 1.9872594e-01


.. parsed-literal::

      11	-3.7271370e-03	 2.3960033e-01	 1.0456443e-02	 2.5017659e-01	[-2.8018232e-02]	 1.9146895e-01


.. parsed-literal::

      12	-2.6302569e-04	 2.3906411e-01	 1.3733584e-02	 2.4929413e-01	[-2.5934236e-02]	 2.0757461e-01
      13	 2.5195744e-03	 2.3848942e-01	 1.6506078e-02	 2.4859745e-01	[-2.4140570e-02]	 1.9667387e-01


.. parsed-literal::

      14	 6.3854843e-03	 2.3762359e-01	 2.0800474e-02	 2.4746005e-01	[-2.0327749e-02]	 2.0297027e-01


.. parsed-literal::

      15	 1.0562519e-01	 2.2460895e-01	 1.2772446e-01	 2.3371592e-01	[ 9.3169786e-02]	 3.2015896e-01


.. parsed-literal::

      16	 1.3630585e-01	 2.1814312e-01	 1.5851354e-01	 2.2809903e-01	[ 1.3259392e-01]	 4.2686820e-01


.. parsed-literal::

      17	 1.8045202e-01	 2.1224788e-01	 2.0356291e-01	 2.2485230e-01	[ 1.7677239e-01]	 2.1631598e-01


.. parsed-literal::

      18	 2.8533622e-01	 2.1301321e-01	 3.1398896e-01	 2.2387031e-01	[ 2.8744148e-01]	 2.2014356e-01
      19	 3.5615408e-01	 2.1175669e-01	 3.8873155e-01	 2.2207923e-01	[ 3.5510758e-01]	 1.9939256e-01


.. parsed-literal::

      20	 4.0206792e-01	 2.0686541e-01	 4.3589803e-01	 2.1547229e-01	[ 4.1165908e-01]	 2.0987844e-01


.. parsed-literal::

      21	 4.5234679e-01	 2.0190820e-01	 4.8693265e-01	 2.1010628e-01	[ 4.6729974e-01]	 2.1954513e-01


.. parsed-literal::

      22	 5.1577971e-01	 1.9688214e-01	 5.5138122e-01	 2.0453898e-01	[ 5.3966837e-01]	 2.1001744e-01


.. parsed-literal::

      23	 6.1686594e-01	 1.9183630e-01	 6.5716217e-01	 1.9979506e-01	[ 6.4830041e-01]	 2.2260070e-01


.. parsed-literal::

      24	 6.4298117e-01	 1.8959482e-01	 6.8466680e-01	 1.9860781e-01	[ 6.7184561e-01]	 3.1823397e-01


.. parsed-literal::

      25	 6.7203717e-01	 1.8867454e-01	 7.1292596e-01	 1.9726846e-01	[ 7.0066335e-01]	 2.1115780e-01
      26	 7.0527384e-01	 1.8901046e-01	 7.4400268e-01	 1.9735311e-01	[ 7.2932903e-01]	 1.7990685e-01


.. parsed-literal::

      27	 7.3441130e-01	 1.9048252e-01	 7.7258475e-01	 2.0004493e-01	[ 7.5498381e-01]	 2.1125007e-01


.. parsed-literal::

      28	 7.5469505e-01	 1.9306048e-01	 7.9263788e-01	 2.0301043e-01	[ 7.7663052e-01]	 3.1732583e-01


.. parsed-literal::

      29	 7.7625752e-01	 1.9303862e-01	 8.1488637e-01	 2.0384919e-01	[ 7.9501960e-01]	 2.0103693e-01


.. parsed-literal::

      30	 8.0646916e-01	 1.9115472e-01	 8.4607665e-01	 2.0517070e-01	[ 8.1357032e-01]	 2.1108627e-01


.. parsed-literal::

      31	 8.3165661e-01	 1.8945635e-01	 8.7223268e-01	 2.0470720e-01	[ 8.3496844e-01]	 2.1087718e-01


.. parsed-literal::

      32	 8.5700207e-01	 1.8733530e-01	 8.9721553e-01	 2.0235918e-01	[ 8.6309350e-01]	 2.1389771e-01


.. parsed-literal::

      33	 8.7383743e-01	 1.8380486e-01	 9.1443228e-01	 1.9868076e-01	[ 8.8218563e-01]	 2.1134591e-01


.. parsed-literal::

      34	 8.9601720e-01	 1.8154117e-01	 9.3708198e-01	 1.9635525e-01	[ 9.0021231e-01]	 2.1423340e-01
      35	 9.1679438e-01	 1.7838624e-01	 9.5883332e-01	 1.9295074e-01	[ 9.2686400e-01]	 1.9999599e-01


.. parsed-literal::

      36	 9.3173449e-01	 1.7561185e-01	 9.7476695e-01	 1.8920758e-01	[ 9.4201786e-01]	 1.8942142e-01


.. parsed-literal::

      37	 9.4590306e-01	 1.7457663e-01	 9.8930036e-01	 1.8736981e-01	[ 9.4995137e-01]	 2.0899487e-01
      38	 9.6695079e-01	 1.7249596e-01	 1.0112539e+00	 1.8502496e-01	[ 9.6200996e-01]	 1.9426131e-01


.. parsed-literal::

      39	 9.8354513e-01	 1.7084160e-01	 1.0282117e+00	 1.8347429e-01	[ 9.6717893e-01]	 1.9191813e-01


.. parsed-literal::

      40	 9.9712366e-01	 1.6902158e-01	 1.0422969e+00	 1.8244134e-01	[ 9.7895158e-01]	 2.1784472e-01


.. parsed-literal::

      41	 1.0098384e+00	 1.6799777e-01	 1.0548968e+00	 1.8148760e-01	[ 9.9073113e-01]	 2.0489359e-01
      42	 1.0246800e+00	 1.6684341e-01	 1.0701805e+00	 1.8077878e-01	[ 1.0010533e+00]	 1.8673873e-01


.. parsed-literal::

      43	 1.0330616e+00	 1.6564753e-01	 1.0797734e+00	 1.7917203e-01	[ 1.0113025e+00]	 2.0358872e-01
      44	 1.0440750e+00	 1.6512382e-01	 1.0905669e+00	 1.7905964e-01	[ 1.0150055e+00]	 1.8432856e-01


.. parsed-literal::

      45	 1.0512569e+00	 1.6478281e-01	 1.0977235e+00	 1.7900297e-01	[ 1.0162244e+00]	 2.1156716e-01


.. parsed-literal::

      46	 1.0622192e+00	 1.6402450e-01	 1.1089635e+00	 1.7813960e-01	[ 1.0210027e+00]	 2.1198797e-01


.. parsed-literal::

      47	 1.0768922e+00	 1.6209728e-01	 1.1239905e+00	 1.7492075e-01	[ 1.0356388e+00]	 2.2047424e-01
      48	 1.0886890e+00	 1.6002554e-01	 1.1366756e+00	 1.7035464e-01	[ 1.0606741e+00]	 1.9195151e-01


.. parsed-literal::

      49	 1.0992575e+00	 1.5829352e-01	 1.1470958e+00	 1.6853664e-01	[ 1.0698635e+00]	 2.1429634e-01


.. parsed-literal::

      50	 1.1103674e+00	 1.5590673e-01	 1.1582416e+00	 1.6570779e-01	[ 1.0827560e+00]	 2.0875621e-01


.. parsed-literal::

      51	 1.1223692e+00	 1.5338643e-01	 1.1705913e+00	 1.6223323e-01	[ 1.0948584e+00]	 2.1877980e-01


.. parsed-literal::

      52	 1.1306762e+00	 1.5101615e-01	 1.1793306e+00	 1.5905709e-01	[ 1.0975434e+00]	 2.1864867e-01


.. parsed-literal::

      53	 1.1390266e+00	 1.5058647e-01	 1.1875659e+00	 1.5892465e-01	[ 1.1064529e+00]	 2.1479964e-01


.. parsed-literal::

      54	 1.1505339e+00	 1.4958999e-01	 1.1994422e+00	 1.5833055e-01	[ 1.1226396e+00]	 2.1462488e-01


.. parsed-literal::

      55	 1.1577299e+00	 1.4928193e-01	 1.2066409e+00	 1.5859830e-01	[ 1.1242887e+00]	 2.1001840e-01
      56	 1.1674677e+00	 1.4846779e-01	 1.2165468e+00	 1.5871810e-01	[ 1.1359905e+00]	 1.8042922e-01


.. parsed-literal::

      57	 1.1755453e+00	 1.4799829e-01	 1.2247693e+00	 1.5880945e-01	[ 1.1449859e+00]	 1.8564320e-01
      58	 1.1823374e+00	 1.4754120e-01	 1.2318770e+00	 1.6017415e-01	[ 1.1491124e+00]	 1.8536687e-01


.. parsed-literal::

      59	 1.1894191e+00	 1.4798076e-01	 1.2390011e+00	 1.6012514e-01	[ 1.1602160e+00]	 2.0878100e-01


.. parsed-literal::

      60	 1.1934267e+00	 1.4751026e-01	 1.2428371e+00	 1.5943932e-01	[ 1.1634143e+00]	 2.1522546e-01


.. parsed-literal::

      61	 1.2002137e+00	 1.4733049e-01	 1.2498063e+00	 1.5916864e-01	[ 1.1658079e+00]	 2.1870208e-01
      62	 1.2090434e+00	 1.4751350e-01	 1.2589693e+00	 1.5955529e-01	  1.1642368e+00 	 1.9941306e-01


.. parsed-literal::

      63	 1.2121459e+00	 1.4813837e-01	 1.2626121e+00	 1.6032801e-01	  1.1650016e+00 	 2.1150994e-01
      64	 1.2228964e+00	 1.4708919e-01	 1.2730528e+00	 1.5918298e-01	[ 1.1722534e+00]	 1.8100762e-01


.. parsed-literal::

      65	 1.2272980e+00	 1.4673919e-01	 1.2774896e+00	 1.5882005e-01	[ 1.1753933e+00]	 1.8568802e-01


.. parsed-literal::

      66	 1.2334615e+00	 1.4629301e-01	 1.2838472e+00	 1.5823401e-01	[ 1.1794593e+00]	 2.1291566e-01
      67	 1.2436390e+00	 1.4579462e-01	 1.2944753e+00	 1.5740473e-01	[ 1.1873580e+00]	 2.0216370e-01


.. parsed-literal::

      68	 1.2453193e+00	 1.4550778e-01	 1.2967159e+00	 1.5686079e-01	  1.1777224e+00 	 2.1057963e-01


.. parsed-literal::

      69	 1.2546997e+00	 1.4482080e-01	 1.3057356e+00	 1.5618127e-01	[ 1.1905190e+00]	 2.1346498e-01
      70	 1.2584672e+00	 1.4435906e-01	 1.3094468e+00	 1.5580894e-01	[ 1.1950552e+00]	 1.7891407e-01


.. parsed-literal::

      71	 1.2634204e+00	 1.4365394e-01	 1.3144074e+00	 1.5527379e-01	[ 1.1988913e+00]	 2.0447779e-01
      72	 1.2685552e+00	 1.4247020e-01	 1.3197035e+00	 1.5464397e-01	  1.1969320e+00 	 1.9932222e-01


.. parsed-literal::

      73	 1.2746757e+00	 1.4219487e-01	 1.3258098e+00	 1.5451097e-01	[ 1.1992383e+00]	 2.1754551e-01


.. parsed-literal::

      74	 1.2803204e+00	 1.4221521e-01	 1.3314720e+00	 1.5467071e-01	  1.1974580e+00 	 2.1460772e-01
      75	 1.2861646e+00	 1.4238270e-01	 1.3375661e+00	 1.5501818e-01	  1.1924417e+00 	 1.9814658e-01


.. parsed-literal::

      76	 1.2948090e+00	 1.4249811e-01	 1.3464561e+00	 1.5549136e-01	  1.1869164e+00 	 2.1278667e-01


.. parsed-literal::

      77	 1.3015366e+00	 1.4234006e-01	 1.3536011e+00	 1.5659685e-01	  1.1722209e+00 	 2.1088719e-01


.. parsed-literal::

      78	 1.3077081e+00	 1.4200700e-01	 1.3595319e+00	 1.5615781e-01	  1.1818799e+00 	 2.1052623e-01


.. parsed-literal::

      79	 1.3125323e+00	 1.4145312e-01	 1.3643230e+00	 1.5572241e-01	  1.1828884e+00 	 2.1430731e-01
      80	 1.3190855e+00	 1.4092057e-01	 1.3710106e+00	 1.5511153e-01	  1.1793318e+00 	 1.8000722e-01


.. parsed-literal::

      81	 1.3236194e+00	 1.4031595e-01	 1.3759094e+00	 1.5496419e-01	  1.1655681e+00 	 3.0328465e-01


.. parsed-literal::

      82	 1.3296573e+00	 1.4008566e-01	 1.3821975e+00	 1.5445007e-01	  1.1599232e+00 	 2.0745683e-01


.. parsed-literal::

      83	 1.3343416e+00	 1.3994954e-01	 1.3870296e+00	 1.5397120e-01	  1.1605189e+00 	 2.1030235e-01


.. parsed-literal::

      84	 1.3404167e+00	 1.3979497e-01	 1.3934015e+00	 1.5340956e-01	  1.1645602e+00 	 2.1495175e-01


.. parsed-literal::

      85	 1.3449510e+00	 1.3923177e-01	 1.3982019e+00	 1.5264670e-01	  1.1716959e+00 	 2.0995665e-01


.. parsed-literal::

      86	 1.3497945e+00	 1.3898336e-01	 1.4028581e+00	 1.5238235e-01	  1.1800850e+00 	 2.1072197e-01


.. parsed-literal::

      87	 1.3553791e+00	 1.3828616e-01	 1.4084363e+00	 1.5182794e-01	  1.1857718e+00 	 2.1247530e-01
      88	 1.3605266e+00	 1.3784910e-01	 1.4136236e+00	 1.5127059e-01	  1.1920029e+00 	 1.9990087e-01


.. parsed-literal::

      89	 1.3662214e+00	 1.3689706e-01	 1.4198290e+00	 1.5028512e-01	  1.1928932e+00 	 2.1798515e-01


.. parsed-literal::

      90	 1.3737735e+00	 1.3670087e-01	 1.4271920e+00	 1.4970716e-01	  1.1932398e+00 	 2.1903419e-01


.. parsed-literal::

      91	 1.3780113e+00	 1.3667214e-01	 1.4314149e+00	 1.4929708e-01	  1.1907087e+00 	 2.1142006e-01


.. parsed-literal::

      92	 1.3828763e+00	 1.3623664e-01	 1.4365584e+00	 1.4854339e-01	  1.1805019e+00 	 2.1506262e-01


.. parsed-literal::

      93	 1.3872677e+00	 1.3600817e-01	 1.4409741e+00	 1.4796479e-01	  1.1745551e+00 	 2.1429539e-01


.. parsed-literal::

      94	 1.3908621e+00	 1.3552335e-01	 1.4445253e+00	 1.4763479e-01	  1.1792116e+00 	 2.1985030e-01
      95	 1.3956657e+00	 1.3456965e-01	 1.4492621e+00	 1.4718592e-01	  1.1822602e+00 	 1.9816828e-01


.. parsed-literal::

      96	 1.3993940e+00	 1.3412974e-01	 1.4529625e+00	 1.4705444e-01	  1.1848994e+00 	 2.1256518e-01
      97	 1.4037409e+00	 1.3347360e-01	 1.4573397e+00	 1.4652249e-01	  1.1844380e+00 	 1.9482803e-01


.. parsed-literal::

      98	 1.4102607e+00	 1.3276738e-01	 1.4639869e+00	 1.4553471e-01	  1.1728502e+00 	 2.1057320e-01


.. parsed-literal::

      99	 1.4146537e+00	 1.3208748e-01	 1.4684026e+00	 1.4481836e-01	  1.1709337e+00 	 2.0923567e-01


.. parsed-literal::

     100	 1.4184387e+00	 1.3199458e-01	 1.4721144e+00	 1.4460284e-01	  1.1707672e+00 	 2.1308923e-01


.. parsed-literal::

     101	 1.4250434e+00	 1.3149281e-01	 1.4788494e+00	 1.4413860e-01	  1.1583067e+00 	 2.1916032e-01


.. parsed-literal::

     102	 1.4294553e+00	 1.3100067e-01	 1.4832425e+00	 1.4370953e-01	  1.1580566e+00 	 2.0619535e-01
     103	 1.4361315e+00	 1.2989179e-01	 1.4900947e+00	 1.4326888e-01	  1.1392277e+00 	 2.0806861e-01


.. parsed-literal::

     104	 1.4390928e+00	 1.2929311e-01	 1.4933093e+00	 1.4300176e-01	  1.1514229e+00 	 2.0558071e-01


.. parsed-literal::

     105	 1.4422874e+00	 1.2934642e-01	 1.4963374e+00	 1.4296723e-01	  1.1505919e+00 	 2.1466780e-01


.. parsed-literal::

     106	 1.4453697e+00	 1.2915337e-01	 1.4994500e+00	 1.4286715e-01	  1.1521483e+00 	 2.1859503e-01


.. parsed-literal::

     107	 1.4487496e+00	 1.2889809e-01	 1.5029432e+00	 1.4272106e-01	  1.1555642e+00 	 2.1391273e-01


.. parsed-literal::

     108	 1.4510864e+00	 1.2829455e-01	 1.5054678e+00	 1.4248250e-01	  1.1679784e+00 	 2.1076918e-01


.. parsed-literal::

     109	 1.4551853e+00	 1.2817959e-01	 1.5095338e+00	 1.4251654e-01	  1.1739594e+00 	 3.1852603e-01


.. parsed-literal::

     110	 1.4574704e+00	 1.2805847e-01	 1.5118037e+00	 1.4253639e-01	  1.1782660e+00 	 2.2110915e-01
     111	 1.4604866e+00	 1.2789510e-01	 1.5148730e+00	 1.4267768e-01	  1.1807217e+00 	 2.1673536e-01


.. parsed-literal::

     112	 1.4626007e+00	 1.2758060e-01	 1.5172265e+00	 1.4299758e-01	  1.1776895e+00 	 2.0998836e-01


.. parsed-literal::

     113	 1.4663068e+00	 1.2762678e-01	 1.5208316e+00	 1.4319609e-01	  1.1744724e+00 	 2.1223593e-01


.. parsed-literal::

     114	 1.4681685e+00	 1.2754568e-01	 1.5226958e+00	 1.4327631e-01	  1.1745916e+00 	 2.0796967e-01


.. parsed-literal::

     115	 1.4713342e+00	 1.2737420e-01	 1.5259270e+00	 1.4359287e-01	  1.1692358e+00 	 2.0983577e-01
     116	 1.4733840e+00	 1.2710773e-01	 1.5280233e+00	 1.4387854e-01	  1.1754043e+00 	 1.8760157e-01


.. parsed-literal::

     117	 1.4762971e+00	 1.2701168e-01	 1.5308545e+00	 1.4394180e-01	  1.1723677e+00 	 1.9893932e-01
     118	 1.4787156e+00	 1.2690932e-01	 1.5332565e+00	 1.4417809e-01	  1.1703712e+00 	 1.8476057e-01


.. parsed-literal::

     119	 1.4807741e+00	 1.2685389e-01	 1.5353412e+00	 1.4427525e-01	  1.1688407e+00 	 2.0996523e-01


.. parsed-literal::

     120	 1.4836782e+00	 1.2687428e-01	 1.5384768e+00	 1.4488264e-01	  1.1576946e+00 	 2.0811987e-01
     121	 1.4868774e+00	 1.2683172e-01	 1.5417207e+00	 1.4456240e-01	  1.1619501e+00 	 1.8040490e-01


.. parsed-literal::

     122	 1.4884870e+00	 1.2681969e-01	 1.5433110e+00	 1.4448947e-01	  1.1614337e+00 	 2.0119190e-01


.. parsed-literal::

     123	 1.4921393e+00	 1.2675891e-01	 1.5470325e+00	 1.4430246e-01	  1.1585189e+00 	 2.1027184e-01


.. parsed-literal::

     124	 1.4937489e+00	 1.2693005e-01	 1.5488076e+00	 1.4440934e-01	  1.1517561e+00 	 2.1304345e-01
     125	 1.4966032e+00	 1.2682558e-01	 1.5516122e+00	 1.4419389e-01	  1.1482535e+00 	 1.9907808e-01


.. parsed-literal::

     126	 1.4985823e+00	 1.2676937e-01	 1.5535964e+00	 1.4413918e-01	  1.1450058e+00 	 2.1651149e-01


.. parsed-literal::

     127	 1.5004243e+00	 1.2673241e-01	 1.5554990e+00	 1.4408989e-01	  1.1393377e+00 	 2.1352625e-01


.. parsed-literal::

     128	 1.5034153e+00	 1.2665102e-01	 1.5586872e+00	 1.4378964e-01	  1.1257044e+00 	 2.0782018e-01


.. parsed-literal::

     129	 1.5042659e+00	 1.2641714e-01	 1.5598806e+00	 1.4379159e-01	  1.1049065e+00 	 2.0867419e-01


.. parsed-literal::

     130	 1.5073117e+00	 1.2640448e-01	 1.5627541e+00	 1.4357551e-01	  1.1169542e+00 	 2.1331215e-01


.. parsed-literal::

     131	 1.5085807e+00	 1.2633530e-01	 1.5640119e+00	 1.4343047e-01	  1.1200610e+00 	 2.1214533e-01


.. parsed-literal::

     132	 1.5107685e+00	 1.2617485e-01	 1.5662345e+00	 1.4318559e-01	  1.1250619e+00 	 2.0539045e-01


.. parsed-literal::

     133	 1.5127210e+00	 1.2605506e-01	 1.5683066e+00	 1.4290129e-01	  1.1271434e+00 	 2.1746063e-01


.. parsed-literal::

     134	 1.5150029e+00	 1.2597199e-01	 1.5705754e+00	 1.4282483e-01	  1.1333984e+00 	 2.1219110e-01
     135	 1.5166748e+00	 1.2595487e-01	 1.5722510e+00	 1.4278939e-01	  1.1362779e+00 	 1.8007541e-01


.. parsed-literal::

     136	 1.5184649e+00	 1.2593126e-01	 1.5741006e+00	 1.4271070e-01	  1.1371061e+00 	 1.9652343e-01


.. parsed-literal::

     137	 1.5196470e+00	 1.2606976e-01	 1.5754597e+00	 1.4272444e-01	  1.1441114e+00 	 2.0984006e-01
     138	 1.5221531e+00	 1.2595121e-01	 1.5779457e+00	 1.4260470e-01	  1.1418179e+00 	 1.9576669e-01


.. parsed-literal::

     139	 1.5237355e+00	 1.2586742e-01	 1.5795553e+00	 1.4250368e-01	  1.1411119e+00 	 2.1045613e-01
     140	 1.5250324e+00	 1.2579188e-01	 1.5808724e+00	 1.4242251e-01	  1.1427615e+00 	 1.9955230e-01


.. parsed-literal::

     141	 1.5272144e+00	 1.2564382e-01	 1.5830257e+00	 1.4243062e-01	  1.1490679e+00 	 2.0798135e-01


.. parsed-literal::

     142	 1.5286801e+00	 1.2564581e-01	 1.5845118e+00	 1.4234010e-01	  1.1572017e+00 	 3.2447886e-01
     143	 1.5309178e+00	 1.2555202e-01	 1.5866802e+00	 1.4265981e-01	  1.1621431e+00 	 1.9507527e-01


.. parsed-literal::

     144	 1.5324088e+00	 1.2549495e-01	 1.5881673e+00	 1.4280454e-01	  1.1667644e+00 	 2.0896125e-01


.. parsed-literal::

     145	 1.5342451e+00	 1.2549548e-01	 1.5900576e+00	 1.4320894e-01	  1.1662468e+00 	 2.1444583e-01


.. parsed-literal::

     146	 1.5359100e+00	 1.2545730e-01	 1.5917583e+00	 1.4323521e-01	  1.1646931e+00 	 2.1922445e-01
     147	 1.5375466e+00	 1.2538636e-01	 1.5934331e+00	 1.4318425e-01	  1.1617809e+00 	 2.1163607e-01


.. parsed-literal::

     148	 1.5396947e+00	 1.2530147e-01	 1.5956292e+00	 1.4328691e-01	  1.1538220e+00 	 2.0655847e-01
     149	 1.5413919e+00	 1.2510430e-01	 1.5973435e+00	 1.4331945e-01	  1.1462592e+00 	 1.9447708e-01


.. parsed-literal::

     150	 1.5431348e+00	 1.2494879e-01	 1.5990285e+00	 1.4324486e-01	  1.1449794e+00 	 2.0675063e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 8s, sys: 1.15 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f03d0672560>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.19 s, sys: 45 ms, total: 2.24 s
    Wall time: 696 ms


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

