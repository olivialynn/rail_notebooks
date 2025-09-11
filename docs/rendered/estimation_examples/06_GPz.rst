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
       1	-3.5842581e-01	 3.2487200e-01	-3.4877108e-01	 3.0310700e-01	[-3.1033635e-01]	 4.6323657e-01


.. parsed-literal::

       2	-2.8438003e-01	 3.1332586e-01	-2.5969474e-01	 2.9325811e-01	[-2.0211296e-01]	 2.3485494e-01


.. parsed-literal::

       3	-2.4015829e-01	 2.9277849e-01	-1.9756100e-01	 2.7525555e-01	[-1.2927643e-01]	 2.7829027e-01
       4	-2.0570262e-01	 2.6807112e-01	-1.6465901e-01	 2.5738181e-01	[-9.6465364e-02]	 1.9101119e-01


.. parsed-literal::

       5	-1.1501449e-01	 2.5969422e-01	-7.9186278e-02	 2.4634815e-01	[-2.1157132e-02]	 1.9052911e-01


.. parsed-literal::

       6	-8.1860315e-02	 2.5435483e-01	-5.0481162e-02	 2.4482874e-01	[-1.0162543e-02]	 2.1699309e-01
       7	-6.3936196e-02	 2.5156151e-01	-3.9158166e-02	 2.3951844e-01	[ 8.5155648e-03]	 2.0539188e-01


.. parsed-literal::

       8	-5.1728326e-02	 2.4955793e-01	-3.0849421e-02	 2.3652238e-01	[ 2.0328065e-02]	 2.0064163e-01


.. parsed-literal::

       9	-3.6838121e-02	 2.4673621e-01	-1.8992108e-02	 2.3340826e-01	[ 3.3162780e-02]	 2.1049237e-01


.. parsed-literal::

      10	-2.4585496e-02	 2.4430224e-01	-8.9622183e-03	 2.3322591e-01	[ 3.8132011e-02]	 2.0648742e-01
      11	-2.0618427e-02	 2.4385148e-01	-6.2166902e-03	 2.3386724e-01	  3.3065947e-02 	 1.7552686e-01


.. parsed-literal::

      12	-1.7040483e-02	 2.4311220e-01	-2.8862955e-03	 2.3352645e-01	  3.6665738e-02 	 2.0888281e-01


.. parsed-literal::

      13	-1.3397657e-02	 2.4233620e-01	 6.8797806e-04	 2.3326016e-01	[ 3.8848574e-02]	 2.0994639e-01


.. parsed-literal::

      14	-9.1887288e-03	 2.4136908e-01	 5.4039080e-03	 2.3287190e-01	[ 4.3573224e-02]	 2.0757747e-01


.. parsed-literal::

      15	 9.5562042e-02	 2.2948100e-01	 1.1752545e-01	 2.2174332e-01	[ 1.4462760e-01]	 3.1678367e-01


.. parsed-literal::

      16	 1.4771250e-01	 2.2472433e-01	 1.7221647e-01	 2.1786796e-01	[ 1.8646474e-01]	 4.4336414e-01


.. parsed-literal::

      17	 1.8448922e-01	 2.2208308e-01	 2.0844463e-01	 2.1689604e-01	[ 2.1424366e-01]	 2.1521926e-01


.. parsed-literal::

      18	 2.5373874e-01	 2.1987238e-01	 2.8062836e-01	 2.1495131e-01	[ 2.8564679e-01]	 2.0724201e-01
      19	 2.9351658e-01	 2.1772777e-01	 3.2687185e-01	 2.1444229e-01	[ 3.2451862e-01]	 2.0178342e-01


.. parsed-literal::

      20	 3.3101570e-01	 2.1588647e-01	 3.6469808e-01	 2.1091943e-01	[ 3.6822015e-01]	 2.1538544e-01
      21	 3.6556136e-01	 2.1361489e-01	 3.9883285e-01	 2.0979638e-01	[ 4.0259886e-01]	 1.7494869e-01


.. parsed-literal::

      22	 4.2565525e-01	 2.1143181e-01	 4.5828050e-01	 2.0649429e-01	[ 4.5710323e-01]	 1.8017578e-01


.. parsed-literal::

      23	 4.9345438e-01	 2.1119257e-01	 5.2710926e-01	 2.0610335e-01	[ 5.1967740e-01]	 2.1433902e-01


.. parsed-literal::

      24	 5.7296906e-01	 2.0849355e-01	 6.0914385e-01	 2.0019287e-01	[ 5.9524843e-01]	 2.1888089e-01


.. parsed-literal::

      25	 6.1161544e-01	 2.0638135e-01	 6.5104622e-01	 1.9738197e-01	[ 6.1525446e-01]	 2.0726538e-01
      26	 6.4409056e-01	 2.0415575e-01	 6.8319924e-01	 1.9537906e-01	[ 6.4221843e-01]	 1.7280865e-01


.. parsed-literal::

      27	 6.7802520e-01	 2.0066614e-01	 7.1643120e-01	 1.9387110e-01	[ 6.7687726e-01]	 2.0248723e-01
      28	 7.2229723e-01	 1.9809009e-01	 7.6054576e-01	 1.9666677e-01	[ 7.1872116e-01]	 2.0970654e-01


.. parsed-literal::

      29	 7.4075513e-01	 1.9717612e-01	 7.7908377e-01	 1.9488732e-01	[ 7.4796299e-01]	 2.0011997e-01
      30	 7.6596663e-01	 1.9553334e-01	 8.0515915e-01	 1.9280813e-01	[ 7.6743408e-01]	 1.8519449e-01


.. parsed-literal::

      31	 7.8715476e-01	 1.9638659e-01	 8.2748793e-01	 1.9305710e-01	[ 7.8496275e-01]	 2.1051717e-01
      32	 8.0775325e-01	 1.9770282e-01	 8.4829975e-01	 1.9270130e-01	[ 8.1137536e-01]	 1.9339943e-01


.. parsed-literal::

      33	 8.2723658e-01	 2.0031038e-01	 8.6784889e-01	 1.9513392e-01	[ 8.4181973e-01]	 2.1269965e-01
      34	 8.5015273e-01	 1.9864661e-01	 8.9145271e-01	 1.9272741e-01	[ 8.6037892e-01]	 1.8495226e-01


.. parsed-literal::

      35	 8.7128276e-01	 1.9662548e-01	 9.1254134e-01	 1.9123505e-01	[ 8.8201450e-01]	 1.9030619e-01
      36	 8.9081467e-01	 1.9562291e-01	 9.3268158e-01	 1.8960419e-01	[ 8.9908693e-01]	 1.7395091e-01


.. parsed-literal::

      37	 9.1432834e-01	 1.9387094e-01	 9.5657611e-01	 1.8772987e-01	[ 9.2121591e-01]	 2.0563364e-01
      38	 9.3588080e-01	 1.9544451e-01	 9.8032380e-01	 1.8877150e-01	[ 9.3979792e-01]	 2.0646763e-01


.. parsed-literal::

      39	 9.5741077e-01	 1.9436277e-01	 1.0019170e+00	 1.8741428e-01	[ 9.5612421e-01]	 2.1262288e-01
      40	 9.7307260e-01	 1.9288596e-01	 1.0178513e+00	 1.8537153e-01	[ 9.6953639e-01]	 1.8261909e-01


.. parsed-literal::

      41	 9.9253395e-01	 1.8986933e-01	 1.0384467e+00	 1.8159328e-01	[ 9.7549609e-01]	 1.9744635e-01


.. parsed-literal::

      42	 1.0079381e+00	 1.8787953e-01	 1.0541750e+00	 1.7897974e-01	[ 9.7875984e-01]	 2.1910667e-01
      43	 1.0181595e+00	 1.8591314e-01	 1.0642582e+00	 1.7767611e-01	[ 9.8750053e-01]	 1.9886374e-01


.. parsed-literal::

      44	 1.0322522e+00	 1.8211680e-01	 1.0785074e+00	 1.7514879e-01	[ 9.9575613e-01]	 2.0338607e-01


.. parsed-literal::

      45	 1.0450316e+00	 1.7932922e-01	 1.0911611e+00	 1.7305109e-01	[ 1.0025815e+00]	 2.0304918e-01


.. parsed-literal::

      46	 1.0616896e+00	 1.7456746e-01	 1.1079929e+00	 1.6998189e-01	  9.9639677e-01 	 2.0534158e-01


.. parsed-literal::

      47	 1.0761314e+00	 1.7110101e-01	 1.1226751e+00	 1.6785229e-01	  9.8470071e-01 	 2.0848465e-01


.. parsed-literal::

      48	 1.0914931e+00	 1.6757825e-01	 1.1389634e+00	 1.6464745e-01	  9.5683718e-01 	 2.0493531e-01


.. parsed-literal::

      49	 1.1059071e+00	 1.6495235e-01	 1.1535636e+00	 1.6270846e-01	  9.6114307e-01 	 2.1895146e-01
      50	 1.1166500e+00	 1.6352103e-01	 1.1646301e+00	 1.6088386e-01	  9.7309362e-01 	 1.9968462e-01


.. parsed-literal::

      51	 1.1356937e+00	 1.5913993e-01	 1.1848543e+00	 1.5649366e-01	  9.7420929e-01 	 2.0839858e-01


.. parsed-literal::

      52	 1.1466380e+00	 1.5593629e-01	 1.1965745e+00	 1.5149863e-01	  9.9703041e-01 	 2.0581865e-01


.. parsed-literal::

      53	 1.1621532e+00	 1.5343634e-01	 1.2116056e+00	 1.5098700e-01	[ 1.0149328e+00]	 2.1153402e-01


.. parsed-literal::

      54	 1.1729400e+00	 1.5183773e-01	 1.2222784e+00	 1.5043750e-01	[ 1.0285977e+00]	 2.1530318e-01


.. parsed-literal::

      55	 1.1880599e+00	 1.4976196e-01	 1.2375667e+00	 1.4895847e-01	[ 1.0540508e+00]	 2.0816350e-01
      56	 1.2000241e+00	 1.4883611e-01	 1.2494094e+00	 1.4914350e-01	[ 1.0801586e+00]	 1.8996358e-01


.. parsed-literal::

      57	 1.2103236e+00	 1.4703018e-01	 1.2599661e+00	 1.4672226e-01	[ 1.1049022e+00]	 2.0653963e-01
      58	 1.2192912e+00	 1.4633046e-01	 1.2688327e+00	 1.4584549e-01	[ 1.1081752e+00]	 1.7534351e-01


.. parsed-literal::

      59	 1.2329638e+00	 1.4532325e-01	 1.2830324e+00	 1.4440112e-01	[ 1.1185575e+00]	 1.8487716e-01
      60	 1.2447636e+00	 1.4428205e-01	 1.2948229e+00	 1.4302802e-01	[ 1.1361211e+00]	 2.0141959e-01


.. parsed-literal::

      61	 1.2587238e+00	 1.4354379e-01	 1.3092193e+00	 1.4273368e-01	[ 1.1483736e+00]	 1.8556499e-01


.. parsed-literal::

      62	 1.2689077e+00	 1.4311906e-01	 1.3191555e+00	 1.4255278e-01	[ 1.1687939e+00]	 2.1727419e-01


.. parsed-literal::

      63	 1.2761628e+00	 1.4238120e-01	 1.3261572e+00	 1.4146653e-01	  1.1630869e+00 	 2.1871400e-01
      64	 1.2866567e+00	 1.4237983e-01	 1.3369856e+00	 1.4230773e-01	  1.1577840e+00 	 1.7470503e-01


.. parsed-literal::

      65	 1.2955856e+00	 1.4209920e-01	 1.3461803e+00	 1.4174384e-01	  1.1532303e+00 	 1.8474841e-01


.. parsed-literal::

      66	 1.3063206e+00	 1.4164056e-01	 1.3571268e+00	 1.4014219e-01	  1.1436514e+00 	 2.0617032e-01


.. parsed-literal::

      67	 1.3150883e+00	 1.4115451e-01	 1.3663593e+00	 1.3911170e-01	  1.1159678e+00 	 2.0570183e-01


.. parsed-literal::

      68	 1.3231699e+00	 1.4110739e-01	 1.3744417e+00	 1.3831751e-01	  1.1107852e+00 	 2.0386410e-01


.. parsed-literal::

      69	 1.3290570e+00	 1.4071111e-01	 1.3804021e+00	 1.3786402e-01	  1.0997964e+00 	 2.0597816e-01
      70	 1.3390138e+00	 1.3991864e-01	 1.3905352e+00	 1.3706124e-01	  1.0717460e+00 	 2.0646644e-01


.. parsed-literal::

      71	 1.3421002e+00	 1.3996993e-01	 1.3942521e+00	 1.3603127e-01	  1.0011547e+00 	 2.1039295e-01


.. parsed-literal::

      72	 1.3528546e+00	 1.3905195e-01	 1.4045549e+00	 1.3621527e-01	  1.0299954e+00 	 2.1579051e-01


.. parsed-literal::

      73	 1.3562588e+00	 1.3871436e-01	 1.4079061e+00	 1.3621545e-01	  1.0358912e+00 	 2.1069646e-01


.. parsed-literal::

      74	 1.3628756e+00	 1.3787055e-01	 1.4146894e+00	 1.3611488e-01	  1.0339916e+00 	 2.2714710e-01
      75	 1.3675572e+00	 1.3736256e-01	 1.4194833e+00	 1.3643465e-01	  1.0399116e+00 	 1.8297100e-01


.. parsed-literal::

      76	 1.3731061e+00	 1.3692715e-01	 1.4250703e+00	 1.3600401e-01	  1.0415922e+00 	 2.1255183e-01


.. parsed-literal::

      77	 1.3788504e+00	 1.3660689e-01	 1.4309355e+00	 1.3577022e-01	  1.0438580e+00 	 2.1421099e-01
      78	 1.3831757e+00	 1.3645647e-01	 1.4353558e+00	 1.3542450e-01	  1.0642612e+00 	 1.8039441e-01


.. parsed-literal::

      79	 1.3893039e+00	 1.3640240e-01	 1.4415801e+00	 1.3597586e-01	  1.0987076e+00 	 2.0527124e-01
      80	 1.3936791e+00	 1.3630379e-01	 1.4458457e+00	 1.3598234e-01	  1.1255357e+00 	 1.8566465e-01


.. parsed-literal::

      81	 1.3984494e+00	 1.3617872e-01	 1.4505420e+00	 1.3601443e-01	  1.1474476e+00 	 2.1622205e-01


.. parsed-literal::

      82	 1.4042239e+00	 1.3595831e-01	 1.4564882e+00	 1.3559231e-01	[ 1.1726597e+00]	 2.0216441e-01


.. parsed-literal::

      83	 1.4095080e+00	 1.3564865e-01	 1.4618597e+00	 1.3444796e-01	[ 1.1967673e+00]	 2.1459651e-01


.. parsed-literal::

      84	 1.4139161e+00	 1.3560167e-01	 1.4663464e+00	 1.3373825e-01	[ 1.2084518e+00]	 2.1315694e-01
      85	 1.4192080e+00	 1.3535032e-01	 1.4717448e+00	 1.3328325e-01	[ 1.2183834e+00]	 1.8590903e-01


.. parsed-literal::

      86	 1.4237791e+00	 1.3515296e-01	 1.4765284e+00	 1.3281312e-01	[ 1.2273578e+00]	 1.9603133e-01
      87	 1.4280184e+00	 1.3494003e-01	 1.4807722e+00	 1.3313279e-01	[ 1.2318892e+00]	 1.7179871e-01


.. parsed-literal::

      88	 1.4330123e+00	 1.3467246e-01	 1.4858686e+00	 1.3347101e-01	[ 1.2364490e+00]	 2.0216107e-01


.. parsed-literal::

      89	 1.4374351e+00	 1.3433075e-01	 1.4904103e+00	 1.3354444e-01	[ 1.2379460e+00]	 2.0238924e-01


.. parsed-literal::

      90	 1.4413564e+00	 1.3388015e-01	 1.4945999e+00	 1.3351915e-01	[ 1.2416146e+00]	 2.1025944e-01
      91	 1.4445696e+00	 1.3371294e-01	 1.4977072e+00	 1.3283683e-01	[ 1.2443310e+00]	 2.0621371e-01


.. parsed-literal::

      92	 1.4467654e+00	 1.3350168e-01	 1.4998210e+00	 1.3230016e-01	  1.2428711e+00 	 2.0245743e-01


.. parsed-literal::

      93	 1.4504051e+00	 1.3323296e-01	 1.5036040e+00	 1.3165331e-01	[ 1.2458871e+00]	 2.0185781e-01


.. parsed-literal::

      94	 1.4551607e+00	 1.3291438e-01	 1.5086306e+00	 1.3135474e-01	[ 1.2511862e+00]	 2.1450448e-01


.. parsed-literal::

      95	 1.4583345e+00	 1.3274136e-01	 1.5117865e+00	 1.3098143e-01	[ 1.2544176e+00]	 2.0697570e-01


.. parsed-literal::

      96	 1.4609325e+00	 1.3263936e-01	 1.5142691e+00	 1.3129497e-01	[ 1.2566057e+00]	 2.0900226e-01
      97	 1.4651880e+00	 1.3226218e-01	 1.5185276e+00	 1.3217811e-01	[ 1.2630179e+00]	 1.9127774e-01


.. parsed-literal::

      98	 1.4695210e+00	 1.3167492e-01	 1.5229692e+00	 1.3311019e-01	[ 1.2649367e+00]	 1.8268251e-01


.. parsed-literal::

      99	 1.4719117e+00	 1.3152773e-01	 1.5254650e+00	 1.3369005e-01	[ 1.2736071e+00]	 3.2667804e-01


.. parsed-literal::

     100	 1.4746079e+00	 1.3113325e-01	 1.5282391e+00	 1.3396250e-01	[ 1.2750152e+00]	 2.0610905e-01
     101	 1.4772166e+00	 1.3077932e-01	 1.5308498e+00	 1.3364850e-01	[ 1.2762302e+00]	 1.9530964e-01


.. parsed-literal::

     102	 1.4794620e+00	 1.3037410e-01	 1.5332670e+00	 1.3322104e-01	  1.2747678e+00 	 2.0110440e-01


.. parsed-literal::

     103	 1.4824341e+00	 1.3024894e-01	 1.5361331e+00	 1.3296953e-01	[ 1.2790727e+00]	 2.0816040e-01


.. parsed-literal::

     104	 1.4841232e+00	 1.3020652e-01	 1.5377758e+00	 1.3273367e-01	[ 1.2808079e+00]	 2.0863557e-01


.. parsed-literal::

     105	 1.4869784e+00	 1.2998913e-01	 1.5406711e+00	 1.3212316e-01	[ 1.2829085e+00]	 2.0629835e-01
     106	 1.4883921e+00	 1.2962400e-01	 1.5423214e+00	 1.3034909e-01	  1.2767353e+00 	 1.9751620e-01


.. parsed-literal::

     107	 1.4918081e+00	 1.2948803e-01	 1.5456671e+00	 1.3040630e-01	  1.2819166e+00 	 2.0948362e-01


.. parsed-literal::

     108	 1.4932549e+00	 1.2937522e-01	 1.5471519e+00	 1.3007563e-01	  1.2812941e+00 	 2.1032643e-01


.. parsed-literal::

     109	 1.4948520e+00	 1.2923067e-01	 1.5488537e+00	 1.2960592e-01	  1.2796147e+00 	 2.0690823e-01


.. parsed-literal::

     110	 1.4967913e+00	 1.2905527e-01	 1.5509125e+00	 1.2913213e-01	  1.2694745e+00 	 2.0847416e-01
     111	 1.4988526e+00	 1.2893750e-01	 1.5530683e+00	 1.2871450e-01	  1.2667382e+00 	 1.7697620e-01


.. parsed-literal::

     112	 1.5000837e+00	 1.2888167e-01	 1.5542711e+00	 1.2894998e-01	  1.2649437e+00 	 1.9947433e-01
     113	 1.5017700e+00	 1.2874330e-01	 1.5560032e+00	 1.2943737e-01	  1.2589569e+00 	 1.8693972e-01


.. parsed-literal::

     114	 1.5037461e+00	 1.2862582e-01	 1.5580703e+00	 1.3012988e-01	  1.2528541e+00 	 2.0261669e-01


.. parsed-literal::

     115	 1.5046359e+00	 1.2832379e-01	 1.5595041e+00	 1.3091996e-01	  1.2238831e+00 	 2.2001624e-01


.. parsed-literal::

     116	 1.5078925e+00	 1.2840750e-01	 1.5624620e+00	 1.3109253e-01	  1.2361085e+00 	 2.0924139e-01


.. parsed-literal::

     117	 1.5089459e+00	 1.2837941e-01	 1.5634686e+00	 1.3084714e-01	  1.2387797e+00 	 2.1439433e-01
     118	 1.5108240e+00	 1.2827341e-01	 1.5653873e+00	 1.3066191e-01	  1.2371233e+00 	 1.7753410e-01


.. parsed-literal::

     119	 1.5118515e+00	 1.2809635e-01	 1.5665822e+00	 1.3052365e-01	  1.2373493e+00 	 2.0625901e-01


.. parsed-literal::

     120	 1.5141362e+00	 1.2797337e-01	 1.5688037e+00	 1.3039308e-01	  1.2330460e+00 	 2.0696330e-01
     121	 1.5154081e+00	 1.2785328e-01	 1.5700867e+00	 1.3048417e-01	  1.2289050e+00 	 1.9805717e-01


.. parsed-literal::

     122	 1.5168018e+00	 1.2770907e-01	 1.5714920e+00	 1.3049823e-01	  1.2254214e+00 	 2.0551896e-01


.. parsed-literal::

     123	 1.5192475e+00	 1.2754917e-01	 1.5739383e+00	 1.3037189e-01	  1.2218988e+00 	 2.0227027e-01


.. parsed-literal::

     124	 1.5202686e+00	 1.2747741e-01	 1.5749819e+00	 1.3034815e-01	  1.2182230e+00 	 3.1736946e-01
     125	 1.5221363e+00	 1.2743449e-01	 1.5768701e+00	 1.3018161e-01	  1.2187153e+00 	 1.7437863e-01


.. parsed-literal::

     126	 1.5233178e+00	 1.2746005e-01	 1.5780601e+00	 1.3026787e-01	  1.2202694e+00 	 2.1021390e-01
     127	 1.5247563e+00	 1.2744797e-01	 1.5796259e+00	 1.3030986e-01	  1.2146785e+00 	 1.6617846e-01


.. parsed-literal::

     128	 1.5259470e+00	 1.2748389e-01	 1.5807879e+00	 1.3063813e-01	  1.2171887e+00 	 2.0561504e-01


.. parsed-literal::

     129	 1.5267177e+00	 1.2741465e-01	 1.5815463e+00	 1.3069553e-01	  1.2156515e+00 	 2.0821595e-01


.. parsed-literal::

     130	 1.5285601e+00	 1.2717675e-01	 1.5834364e+00	 1.3074839e-01	  1.2092017e+00 	 2.1307492e-01


.. parsed-literal::

     131	 1.5293218e+00	 1.2700174e-01	 1.5843367e+00	 1.3064062e-01	  1.2008362e+00 	 2.0787120e-01


.. parsed-literal::

     132	 1.5307732e+00	 1.2696758e-01	 1.5857158e+00	 1.3051835e-01	  1.2024555e+00 	 2.1511960e-01


.. parsed-literal::

     133	 1.5319601e+00	 1.2691141e-01	 1.5869039e+00	 1.3027813e-01	  1.2022372e+00 	 2.0128417e-01
     134	 1.5328771e+00	 1.2684801e-01	 1.5878345e+00	 1.3005180e-01	  1.2007374e+00 	 1.9891310e-01


.. parsed-literal::

     135	 1.5351052e+00	 1.2665798e-01	 1.5901265e+00	 1.2958380e-01	  1.1930048e+00 	 2.0373321e-01


.. parsed-literal::

     136	 1.5363556e+00	 1.2651812e-01	 1.5914466e+00	 1.2917334e-01	  1.1892463e+00 	 3.1922174e-01


.. parsed-literal::

     137	 1.5375629e+00	 1.2641969e-01	 1.5926661e+00	 1.2905726e-01	  1.1851548e+00 	 2.0812845e-01
     138	 1.5385006e+00	 1.2631230e-01	 1.5936403e+00	 1.2900284e-01	  1.1811292e+00 	 2.0092034e-01


.. parsed-literal::

     139	 1.5396895e+00	 1.2616111e-01	 1.5949140e+00	 1.2878336e-01	  1.1751554e+00 	 2.0005822e-01
     140	 1.5411347e+00	 1.2600270e-01	 1.5964245e+00	 1.2857083e-01	  1.1686360e+00 	 1.7708945e-01


.. parsed-literal::

     141	 1.5424457e+00	 1.2592629e-01	 1.5977578e+00	 1.2837383e-01	  1.1672298e+00 	 2.0829320e-01


.. parsed-literal::

     142	 1.5434624e+00	 1.2588229e-01	 1.5988144e+00	 1.2804716e-01	  1.1673872e+00 	 2.1325445e-01


.. parsed-literal::

     143	 1.5443484e+00	 1.2593809e-01	 1.5996419e+00	 1.2815648e-01	  1.1701862e+00 	 2.0321918e-01


.. parsed-literal::

     144	 1.5449583e+00	 1.2595076e-01	 1.6002239e+00	 1.2812804e-01	  1.1701851e+00 	 2.1055627e-01
     145	 1.5460735e+00	 1.2593709e-01	 1.6013237e+00	 1.2806238e-01	  1.1675521e+00 	 1.8008804e-01


.. parsed-literal::

     146	 1.5467671e+00	 1.2586916e-01	 1.6020764e+00	 1.2782397e-01	  1.1625366e+00 	 2.1767712e-01


.. parsed-literal::

     147	 1.5478524e+00	 1.2581340e-01	 1.6031346e+00	 1.2774129e-01	  1.1593112e+00 	 2.0726919e-01
     148	 1.5486294e+00	 1.2574203e-01	 1.6039264e+00	 1.2763422e-01	  1.1553289e+00 	 1.9973826e-01


.. parsed-literal::

     149	 1.5491434e+00	 1.2569489e-01	 1.6044712e+00	 1.2754094e-01	  1.1514726e+00 	 2.0089269e-01


.. parsed-literal::

     150	 1.5500772e+00	 1.2565647e-01	 1.6055366e+00	 1.2746456e-01	  1.1455032e+00 	 2.1295714e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe8d4a75a80>



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
    CPU times: user 2.05 s, sys: 50.9 ms, total: 2.1 s
    Wall time: 630 ms


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

