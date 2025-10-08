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
       1	-3.4447569e-01	 3.2088995e-01	-3.3481809e-01	 3.1897250e-01	[-3.3126075e-01]	 4.6445990e-01


.. parsed-literal::

       2	-2.7326060e-01	 3.1002416e-01	-2.4890903e-01	 3.0805992e-01	[-2.4270068e-01]	 2.3283219e-01


.. parsed-literal::

       3	-2.2782162e-01	 2.8844451e-01	-1.8409808e-01	 2.8736561e-01	[-1.8010894e-01]	 2.7841306e-01
       4	-1.9721698e-01	 2.6496596e-01	-1.5592416e-01	 2.6767473e-01	[-1.6908150e-01]	 1.7257857e-01


.. parsed-literal::

       5	-1.0411216e-01	 2.5731897e-01	-6.9202374e-02	 2.5753932e-01	[-7.0568564e-02]	 2.0649242e-01
       6	-7.1393860e-02	 2.5168236e-01	-4.0620382e-02	 2.5203606e-01	[-4.1619030e-02]	 1.8259406e-01


.. parsed-literal::

       7	-5.3436379e-02	 2.4901530e-01	-2.9288355e-02	 2.4976188e-01	[-3.1389944e-02]	 2.1004415e-01


.. parsed-literal::

       8	-4.1424115e-02	 2.4705031e-01	-2.1049551e-02	 2.4739253e-01	[-2.1627569e-02]	 2.1249866e-01
       9	-2.7442727e-02	 2.4443928e-01	-9.9436161e-03	 2.4403887e-01	[-7.7100318e-03]	 2.0145416e-01


.. parsed-literal::

      10	-1.6474955e-02	 2.4222561e-01	-1.1203582e-03	 2.4204129e-01	[ 1.3065187e-04]	 2.0935130e-01
      11	-1.1355035e-02	 2.4163462e-01	 2.8238668e-03	 2.4165417e-01	[ 3.4828614e-03]	 1.7645574e-01


.. parsed-literal::

      12	-8.6045698e-03	 2.4106693e-01	 5.4051049e-03	 2.4108482e-01	[ 6.0980424e-03]	 2.1735454e-01


.. parsed-literal::

      13	-4.6042144e-03	 2.4025636e-01	 9.4501340e-03	 2.4026062e-01	[ 1.0132096e-02]	 2.1936178e-01


.. parsed-literal::

      14	 1.3628204e-01	 2.2360179e-01	 1.5930247e-01	 2.2565047e-01	[ 1.5425080e-01]	 4.4577026e-01


.. parsed-literal::

      15	 1.7610621e-01	 2.1894823e-01	 1.9953986e-01	 2.2158912e-01	[ 2.0389692e-01]	 3.2040119e-01


.. parsed-literal::

      16	 2.5427806e-01	 2.1701650e-01	 2.8307199e-01	 2.1875067e-01	[ 2.9440444e-01]	 2.2615528e-01


.. parsed-literal::

      17	 2.9198690e-01	 2.1087729e-01	 3.2438082e-01	 2.1280800e-01	[ 3.4297703e-01]	 2.1858096e-01


.. parsed-literal::

      18	 3.1971980e-01	 2.0944980e-01	 3.5236597e-01	 2.1273853e-01	[ 3.6421617e-01]	 2.1983576e-01


.. parsed-literal::

      19	 3.6963090e-01	 2.0648040e-01	 4.0265180e-01	 2.0917042e-01	[ 4.1361477e-01]	 2.0641780e-01


.. parsed-literal::

      20	 4.3244496e-01	 2.0530744e-01	 4.6460318e-01	 2.0850491e-01	[ 4.7523470e-01]	 2.0414734e-01


.. parsed-literal::

      21	 5.3135006e-01	 2.0598782e-01	 5.6569133e-01	 2.1159270e-01	[ 5.7858545e-01]	 2.2158813e-01


.. parsed-literal::

      22	 5.5478064e-01	 2.0808914e-01	 5.9731161e-01	 2.1465888e-01	[ 5.8629554e-01]	 2.0569015e-01


.. parsed-literal::

      23	 6.2151860e-01	 2.0202770e-01	 6.6012807e-01	 2.0886479e-01	[ 6.6335770e-01]	 2.1296644e-01
      24	 6.4738339e-01	 1.9804058e-01	 6.8482984e-01	 2.0426101e-01	[ 6.8679323e-01]	 1.8977332e-01


.. parsed-literal::

      25	 6.8975801e-01	 1.9440231e-01	 7.2526968e-01	 2.0035643e-01	[ 7.2768671e-01]	 2.1828270e-01


.. parsed-literal::

      26	 7.1413522e-01	 1.9478945e-01	 7.4987501e-01	 2.0228099e-01	[ 7.4373042e-01]	 2.2025156e-01
      27	 7.4662505e-01	 1.9224466e-01	 7.8425507e-01	 1.9984616e-01	[ 7.8301495e-01]	 2.0740342e-01


.. parsed-literal::

      28	 7.6785991e-01	 1.9179124e-01	 8.0504901e-01	 1.9940656e-01	[ 8.0420022e-01]	 2.1307111e-01
      29	 7.8486625e-01	 1.9012247e-01	 8.2244138e-01	 1.9705563e-01	[ 8.2023902e-01]	 1.9166780e-01


.. parsed-literal::

      30	 8.2745098e-01	 1.8853803e-01	 8.6725056e-01	 1.9476389e-01	[ 8.6020091e-01]	 2.1571064e-01


.. parsed-literal::

      31	 8.6328931e-01	 1.8589815e-01	 9.0378735e-01	 1.9212636e-01	[ 8.9651751e-01]	 2.2368383e-01


.. parsed-literal::

      32	 8.9119156e-01	 1.8311134e-01	 9.3167468e-01	 1.9018071e-01	[ 9.2403032e-01]	 2.1724319e-01


.. parsed-literal::

      33	 9.1983273e-01	 1.8099195e-01	 9.6176036e-01	 1.9009160e-01	[ 9.5206980e-01]	 2.1665478e-01
      34	 9.2869034e-01	 1.7876611e-01	 9.7107765e-01	 1.8647514e-01	[ 9.6131705e-01]	 1.9991016e-01


.. parsed-literal::

      35	 9.3999416e-01	 1.7776532e-01	 9.8209829e-01	 1.8562743e-01	[ 9.7311362e-01]	 1.7779517e-01


.. parsed-literal::

      36	 9.5236401e-01	 1.7555737e-01	 9.9487699e-01	 1.8330765e-01	[ 9.8470532e-01]	 2.1205378e-01
      37	 9.6156210e-01	 1.7465438e-01	 1.0045133e+00	 1.8242306e-01	[ 9.9288330e-01]	 1.9841862e-01


.. parsed-literal::

      38	 9.7465603e-01	 1.7297379e-01	 1.0182758e+00	 1.8219798e-01	[ 1.0025619e+00]	 2.1400070e-01


.. parsed-literal::

      39	 9.8851166e-01	 1.7186206e-01	 1.0325215e+00	 1.8155571e-01	[ 1.0120706e+00]	 2.0642400e-01


.. parsed-literal::

      40	 1.0031250e+00	 1.7182576e-01	 1.0475646e+00	 1.8288489e-01	[ 1.0237178e+00]	 2.0511079e-01


.. parsed-literal::

      41	 1.0149517e+00	 1.7096888e-01	 1.0596113e+00	 1.8237366e-01	[ 1.0311481e+00]	 2.0479155e-01


.. parsed-literal::

      42	 1.0272659e+00	 1.7056281e-01	 1.0721160e+00	 1.8241284e-01	[ 1.0379378e+00]	 2.0974159e-01


.. parsed-literal::

      43	 1.0353451e+00	 1.6997007e-01	 1.0812721e+00	 1.8304517e-01	[ 1.0422455e+00]	 2.1597171e-01


.. parsed-literal::

      44	 1.0462044e+00	 1.6916434e-01	 1.0920463e+00	 1.8194203e-01	[ 1.0533738e+00]	 2.0194578e-01
      45	 1.0528090e+00	 1.6850962e-01	 1.0987636e+00	 1.8081408e-01	[ 1.0608202e+00]	 1.9761801e-01


.. parsed-literal::

      46	 1.0647832e+00	 1.6785087e-01	 1.1109177e+00	 1.7989280e-01	[ 1.0731857e+00]	 1.9902015e-01


.. parsed-literal::

      47	 1.0749136e+00	 1.6648308e-01	 1.1221763e+00	 1.7883920e-01	[ 1.0790996e+00]	 2.1784258e-01


.. parsed-literal::

      48	 1.0885125e+00	 1.6587130e-01	 1.1354058e+00	 1.7908246e-01	[ 1.0901103e+00]	 2.1314955e-01
      49	 1.0942686e+00	 1.6514549e-01	 1.1409191e+00	 1.7856907e-01	[ 1.0959046e+00]	 1.9663811e-01


.. parsed-literal::

      50	 1.1047839e+00	 1.6374833e-01	 1.1516219e+00	 1.7806853e-01	[ 1.1057768e+00]	 2.0685387e-01


.. parsed-literal::

      51	 1.1146883e+00	 1.6155072e-01	 1.1617336e+00	 1.7511231e-01	[ 1.1170255e+00]	 2.0257187e-01
      52	 1.1257543e+00	 1.5958596e-01	 1.1732060e+00	 1.7357088e-01	[ 1.1264129e+00]	 2.0599174e-01


.. parsed-literal::

      53	 1.1367156e+00	 1.5807084e-01	 1.1846920e+00	 1.7176581e-01	[ 1.1310851e+00]	 2.2054672e-01
      54	 1.1502885e+00	 1.5600050e-01	 1.1989519e+00	 1.6998664e-01	[ 1.1420132e+00]	 1.7752337e-01


.. parsed-literal::

      55	 1.1623555e+00	 1.5548655e-01	 1.2110993e+00	 1.6947703e-01	[ 1.1433278e+00]	 2.1807718e-01


.. parsed-literal::

      56	 1.1725098e+00	 1.5506124e-01	 1.2213687e+00	 1.6896337e-01	[ 1.1502886e+00]	 2.1538615e-01
      57	 1.1839091e+00	 1.5447287e-01	 1.2331716e+00	 1.6840531e-01	[ 1.1595552e+00]	 1.8240643e-01


.. parsed-literal::

      58	 1.1926162e+00	 1.5301239e-01	 1.2419564e+00	 1.6741276e-01	[ 1.1653281e+00]	 1.7421746e-01


.. parsed-literal::

      59	 1.2018188e+00	 1.5173292e-01	 1.2511300e+00	 1.6603393e-01	[ 1.1759235e+00]	 2.0738769e-01
      60	 1.2103605e+00	 1.5056744e-01	 1.2600002e+00	 1.6543636e-01	[ 1.1810024e+00]	 2.0078897e-01


.. parsed-literal::

      61	 1.2208931e+00	 1.4928188e-01	 1.2708272e+00	 1.6386937e-01	[ 1.1929125e+00]	 2.1357846e-01


.. parsed-literal::

      62	 1.2291454e+00	 1.4927585e-01	 1.2794537e+00	 1.6469921e-01	[ 1.1969788e+00]	 2.2062254e-01


.. parsed-literal::

      63	 1.2368149e+00	 1.4905153e-01	 1.2871052e+00	 1.6463387e-01	[ 1.2037690e+00]	 2.1552134e-01


.. parsed-literal::

      64	 1.2452569e+00	 1.4888863e-01	 1.2958423e+00	 1.6451929e-01	[ 1.2098100e+00]	 2.1435547e-01
      65	 1.2531080e+00	 1.4842168e-01	 1.3036605e+00	 1.6453918e-01	[ 1.2170822e+00]	 1.7645335e-01


.. parsed-literal::

      66	 1.2656764e+00	 1.4839138e-01	 1.3171542e+00	 1.6514564e-01	  1.2158655e+00 	 2.0608902e-01


.. parsed-literal::

      67	 1.2747067e+00	 1.4674870e-01	 1.3257810e+00	 1.6432384e-01	[ 1.2283304e+00]	 2.1149492e-01
      68	 1.2809097e+00	 1.4660895e-01	 1.3316167e+00	 1.6370314e-01	[ 1.2362132e+00]	 1.9278383e-01


.. parsed-literal::

      69	 1.2867364e+00	 1.4665929e-01	 1.3376797e+00	 1.6385984e-01	  1.2356122e+00 	 1.7841625e-01


.. parsed-literal::

      70	 1.2943609e+00	 1.4573277e-01	 1.3458298e+00	 1.6314131e-01	  1.2325922e+00 	 2.1407223e-01
      71	 1.3018202e+00	 1.4454026e-01	 1.3532875e+00	 1.6218003e-01	  1.2319825e+00 	 1.9159126e-01


.. parsed-literal::

      72	 1.3081613e+00	 1.4338543e-01	 1.3597146e+00	 1.6162605e-01	  1.2328452e+00 	 2.0801759e-01


.. parsed-literal::

      73	 1.3136049e+00	 1.4282923e-01	 1.3651803e+00	 1.6083285e-01	[ 1.2389961e+00]	 2.0956373e-01


.. parsed-literal::

      74	 1.3210966e+00	 1.4262750e-01	 1.3727665e+00	 1.6056204e-01	[ 1.2434962e+00]	 2.2036004e-01


.. parsed-literal::

      75	 1.3290826e+00	 1.4176386e-01	 1.3811517e+00	 1.5964261e-01	[ 1.2475016e+00]	 2.1822906e-01


.. parsed-literal::

      76	 1.3357837e+00	 1.4135945e-01	 1.3882652e+00	 1.5936798e-01	  1.2466497e+00 	 2.1799970e-01


.. parsed-literal::

      77	 1.3412216e+00	 1.4093342e-01	 1.3936918e+00	 1.5926563e-01	  1.2474378e+00 	 2.1604657e-01


.. parsed-literal::

      78	 1.3459432e+00	 1.4036858e-01	 1.3984865e+00	 1.5888415e-01	[ 1.2490280e+00]	 2.1086097e-01


.. parsed-literal::

      79	 1.3527443e+00	 1.3961314e-01	 1.4055317e+00	 1.5863233e-01	  1.2484485e+00 	 2.1393943e-01


.. parsed-literal::

      80	 1.3583687e+00	 1.3952179e-01	 1.4112175e+00	 1.5817648e-01	[ 1.2545647e+00]	 2.1258998e-01


.. parsed-literal::

      81	 1.3628727e+00	 1.3921065e-01	 1.4156243e+00	 1.5767331e-01	[ 1.2589223e+00]	 2.1126962e-01
      82	 1.3687459e+00	 1.3863570e-01	 1.4216727e+00	 1.5725093e-01	[ 1.2597107e+00]	 1.9861197e-01


.. parsed-literal::

      83	 1.3734680e+00	 1.3807541e-01	 1.4265309e+00	 1.5714102e-01	[ 1.2611443e+00]	 2.0987630e-01
      84	 1.3784215e+00	 1.3733391e-01	 1.4315530e+00	 1.5693109e-01	[ 1.2631035e+00]	 2.0182943e-01


.. parsed-literal::

      85	 1.3821681e+00	 1.3691572e-01	 1.4353085e+00	 1.5716018e-01	  1.2613976e+00 	 2.0754886e-01


.. parsed-literal::

      86	 1.3862958e+00	 1.3633975e-01	 1.4395735e+00	 1.5726591e-01	  1.2571075e+00 	 2.0701885e-01
      87	 1.3906961e+00	 1.3597277e-01	 1.4440863e+00	 1.5769150e-01	  1.2567419e+00 	 1.9848847e-01


.. parsed-literal::

      88	 1.3949800e+00	 1.3571656e-01	 1.4484583e+00	 1.5764069e-01	  1.2598016e+00 	 1.9807959e-01


.. parsed-literal::

      89	 1.3991706e+00	 1.3554590e-01	 1.4526961e+00	 1.5766611e-01	  1.2618634e+00 	 2.0372128e-01


.. parsed-literal::

      90	 1.4022343e+00	 1.3544430e-01	 1.4558036e+00	 1.5786744e-01	[ 1.2631594e+00]	 2.0657516e-01


.. parsed-literal::

      91	 1.4064143e+00	 1.3528862e-01	 1.4600841e+00	 1.5820215e-01	[ 1.2632948e+00]	 2.1653128e-01
      92	 1.4093652e+00	 1.3492173e-01	 1.4631899e+00	 1.5885930e-01	  1.2610009e+00 	 1.9162583e-01


.. parsed-literal::

      93	 1.4126414e+00	 1.3468749e-01	 1.4664005e+00	 1.5855239e-01	  1.2628040e+00 	 1.9909000e-01


.. parsed-literal::

      94	 1.4149956e+00	 1.3434534e-01	 1.4687296e+00	 1.5817011e-01	[ 1.2636721e+00]	 2.0682406e-01


.. parsed-literal::

      95	 1.4175969e+00	 1.3391788e-01	 1.4713296e+00	 1.5768269e-01	[ 1.2638067e+00]	 2.0322132e-01
      96	 1.4219032e+00	 1.3305452e-01	 1.4757297e+00	 1.5682230e-01	  1.2621740e+00 	 1.8690538e-01


.. parsed-literal::

      97	 1.4233585e+00	 1.3238881e-01	 1.4774602e+00	 1.5619759e-01	  1.2468706e+00 	 2.1175480e-01


.. parsed-literal::

      98	 1.4276975e+00	 1.3217924e-01	 1.4816423e+00	 1.5610910e-01	  1.2546954e+00 	 2.1230435e-01


.. parsed-literal::

      99	 1.4295672e+00	 1.3214186e-01	 1.4835215e+00	 1.5624589e-01	  1.2552606e+00 	 2.1273947e-01
     100	 1.4320244e+00	 1.3174991e-01	 1.4861814e+00	 1.5595492e-01	  1.2487106e+00 	 1.8213582e-01


.. parsed-literal::

     101	 1.4350014e+00	 1.3167238e-01	 1.4892406e+00	 1.5603632e-01	  1.2458506e+00 	 1.9097733e-01


.. parsed-literal::

     102	 1.4373832e+00	 1.3142119e-01	 1.4917002e+00	 1.5585660e-01	  1.2427571e+00 	 2.1823597e-01
     103	 1.4410918e+00	 1.3093367e-01	 1.4955322e+00	 1.5542458e-01	  1.2407139e+00 	 1.9413972e-01


.. parsed-literal::

     104	 1.4429011e+00	 1.3064558e-01	 1.4974232e+00	 1.5510935e-01	  1.2424776e+00 	 3.2677794e-01
     105	 1.4451972e+00	 1.3033173e-01	 1.4997244e+00	 1.5479172e-01	  1.2460648e+00 	 1.8020344e-01


.. parsed-literal::

     106	 1.4476111e+00	 1.2997129e-01	 1.5021228e+00	 1.5424590e-01	  1.2511464e+00 	 2.1410584e-01


.. parsed-literal::

     107	 1.4490994e+00	 1.2941385e-01	 1.5036499e+00	 1.5363995e-01	  1.2566902e+00 	 2.1814084e-01


.. parsed-literal::

     108	 1.4507815e+00	 1.2935414e-01	 1.5052518e+00	 1.5345734e-01	  1.2563665e+00 	 2.0357680e-01


.. parsed-literal::

     109	 1.4530222e+00	 1.2899718e-01	 1.5074996e+00	 1.5286814e-01	  1.2522292e+00 	 2.1764517e-01
     110	 1.4547367e+00	 1.2865444e-01	 1.5092131e+00	 1.5230635e-01	  1.2503859e+00 	 1.8630910e-01


.. parsed-literal::

     111	 1.4570729e+00	 1.2797389e-01	 1.5115932e+00	 1.5141928e-01	  1.2427598e+00 	 1.9683409e-01


.. parsed-literal::

     112	 1.4593457e+00	 1.2755955e-01	 1.5138155e+00	 1.5038310e-01	  1.2455517e+00 	 2.1023941e-01


.. parsed-literal::

     113	 1.4609198e+00	 1.2751255e-01	 1.5153384e+00	 1.5045514e-01	  1.2479069e+00 	 2.1295571e-01
     114	 1.4627973e+00	 1.2733065e-01	 1.5172043e+00	 1.5025007e-01	  1.2495778e+00 	 1.8175316e-01


.. parsed-literal::

     115	 1.4641614e+00	 1.2727162e-01	 1.5185939e+00	 1.5023074e-01	  1.2499852e+00 	 2.1555638e-01
     116	 1.4663099e+00	 1.2702547e-01	 1.5207852e+00	 1.4976168e-01	  1.2482395e+00 	 1.8959665e-01


.. parsed-literal::

     117	 1.4699662e+00	 1.2657168e-01	 1.5244921e+00	 1.4905551e-01	  1.2443679e+00 	 2.1824074e-01


.. parsed-literal::

     118	 1.4714438e+00	 1.2625796e-01	 1.5260242e+00	 1.4839716e-01	  1.2486988e+00 	 3.2989693e-01


.. parsed-literal::

     119	 1.4734290e+00	 1.2602536e-01	 1.5280220e+00	 1.4810174e-01	  1.2482999e+00 	 2.1248555e-01


.. parsed-literal::

     120	 1.4753340e+00	 1.2583409e-01	 1.5299545e+00	 1.4791832e-01	  1.2504860e+00 	 2.0950866e-01


.. parsed-literal::

     121	 1.4770298e+00	 1.2568710e-01	 1.5317703e+00	 1.4767427e-01	  1.2498994e+00 	 2.0473146e-01


.. parsed-literal::

     122	 1.4790651e+00	 1.2555543e-01	 1.5338033e+00	 1.4763574e-01	  1.2533201e+00 	 2.0893216e-01


.. parsed-literal::

     123	 1.4807201e+00	 1.2550519e-01	 1.5354935e+00	 1.4769569e-01	  1.2525201e+00 	 2.1207881e-01


.. parsed-literal::

     124	 1.4823662e+00	 1.2540602e-01	 1.5372209e+00	 1.4765922e-01	  1.2489679e+00 	 2.0939875e-01


.. parsed-literal::

     125	 1.4842999e+00	 1.2527258e-01	 1.5393107e+00	 1.4778293e-01	  1.2431157e+00 	 2.1904063e-01
     126	 1.4862217e+00	 1.2503961e-01	 1.5412994e+00	 1.4741235e-01	  1.2407492e+00 	 1.9942570e-01


.. parsed-literal::

     127	 1.4875046e+00	 1.2493569e-01	 1.5425570e+00	 1.4725919e-01	  1.2432573e+00 	 2.0444417e-01


.. parsed-literal::

     128	 1.4892095e+00	 1.2472049e-01	 1.5442972e+00	 1.4695496e-01	  1.2462026e+00 	 2.0804262e-01
     129	 1.4906286e+00	 1.2449601e-01	 1.5458491e+00	 1.4664799e-01	  1.2533543e+00 	 1.8942022e-01


.. parsed-literal::

     130	 1.4922543e+00	 1.2444238e-01	 1.5474194e+00	 1.4658164e-01	  1.2542345e+00 	 1.9528699e-01
     131	 1.4933502e+00	 1.2441245e-01	 1.5484991e+00	 1.4656124e-01	  1.2534422e+00 	 1.7271996e-01


.. parsed-literal::

     132	 1.4945935e+00	 1.2434037e-01	 1.5497490e+00	 1.4650382e-01	  1.2520371e+00 	 2.0296574e-01


.. parsed-literal::

     133	 1.4965155e+00	 1.2414323e-01	 1.5517442e+00	 1.4623097e-01	  1.2508863e+00 	 2.0435691e-01


.. parsed-literal::

     134	 1.4982714e+00	 1.2399663e-01	 1.5535361e+00	 1.4613149e-01	  1.2442098e+00 	 2.1845460e-01
     135	 1.4992419e+00	 1.2397951e-01	 1.5544503e+00	 1.4615942e-01	  1.2476867e+00 	 1.7651248e-01


.. parsed-literal::

     136	 1.5002941e+00	 1.2388552e-01	 1.5555310e+00	 1.4596673e-01	  1.2494126e+00 	 2.1182561e-01


.. parsed-literal::

     137	 1.5019233e+00	 1.2371424e-01	 1.5572225e+00	 1.4578095e-01	  1.2528481e+00 	 2.0709205e-01
     138	 1.5038707e+00	 1.2350125e-01	 1.5592781e+00	 1.4530761e-01	  1.2524248e+00 	 1.7976284e-01


.. parsed-literal::

     139	 1.5056405e+00	 1.2324158e-01	 1.5610433e+00	 1.4506163e-01	  1.2547549e+00 	 2.1185827e-01
     140	 1.5070934e+00	 1.2303515e-01	 1.5624818e+00	 1.4494155e-01	  1.2572550e+00 	 2.0570922e-01


.. parsed-literal::

     141	 1.5082672e+00	 1.2275133e-01	 1.5636613e+00	 1.4461596e-01	  1.2565459e+00 	 2.0761418e-01


.. parsed-literal::

     142	 1.5096112e+00	 1.2260965e-01	 1.5649746e+00	 1.4461667e-01	  1.2563001e+00 	 2.0865512e-01
     143	 1.5111261e+00	 1.2235652e-01	 1.5665127e+00	 1.4450320e-01	  1.2559300e+00 	 1.8620324e-01


.. parsed-literal::

     144	 1.5123248e+00	 1.2218049e-01	 1.5677877e+00	 1.4439103e-01	  1.2524928e+00 	 2.1370864e-01


.. parsed-literal::

     145	 1.5138727e+00	 1.2197187e-01	 1.5694409e+00	 1.4426049e-01	  1.2494042e+00 	 2.1472859e-01


.. parsed-literal::

     146	 1.5153017e+00	 1.2184639e-01	 1.5709801e+00	 1.4407318e-01	  1.2458745e+00 	 2.0451975e-01
     147	 1.5166153e+00	 1.2174215e-01	 1.5723614e+00	 1.4399287e-01	  1.2344724e+00 	 1.8000174e-01


.. parsed-literal::

     148	 1.5176471e+00	 1.2170777e-01	 1.5733760e+00	 1.4406447e-01	  1.2330183e+00 	 2.0927906e-01


.. parsed-literal::

     149	 1.5187140e+00	 1.2161681e-01	 1.5744061e+00	 1.4400897e-01	  1.2286661e+00 	 2.0556688e-01


.. parsed-literal::

     150	 1.5200919e+00	 1.2143896e-01	 1.5757832e+00	 1.4400299e-01	  1.2218736e+00 	 2.0955801e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.16 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5a78c2ca60>



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
    CPU times: user 1.77 s, sys: 40 ms, total: 1.81 s
    Wall time: 570 ms


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

