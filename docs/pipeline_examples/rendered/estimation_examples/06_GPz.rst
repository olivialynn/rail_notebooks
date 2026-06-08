GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

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
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

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
       1	-3.2236243e-01	 3.1415732e-01	-3.1181424e-01	 3.4021945e-01	[-3.6348101e-01]	 4.9250603e-01


.. parsed-literal::

       2	-2.5029200e-01	 3.0242242e-01	-2.2407760e-01	 3.2799284e-01	[-3.0753647e-01]	 2.4580002e-01


.. parsed-literal::

       3	-2.0720475e-01	 2.8340948e-01	-1.6420619e-01	 3.1114473e-01	[-2.8367195e-01]	 2.8871083e-01


.. parsed-literal::

       4	-1.6725894e-01	 2.6582309e-01	-1.1752380e-01	 2.9356922e-01	[-2.6038778e-01]	 3.3273220e-01


.. parsed-literal::

       5	-1.1549377e-01	 2.5183481e-01	-8.2681348e-02	 2.7607677e-01	[-2.3172179e-01]	 2.0962930e-01


.. parsed-literal::

       6	-5.2117259e-02	 2.4726128e-01	-2.1494099e-02	 2.7177925e-01	[-1.2771909e-01]	 2.1009207e-01


.. parsed-literal::

       7	-3.3872640e-02	 2.4358533e-01	-7.7299083e-03	 2.6782292e-01	[-1.0599789e-01]	 2.2254419e-01


.. parsed-literal::

       8	-2.1666615e-02	 2.4164930e-01	 9.7067728e-04	 2.6580228e-01	[-1.0075740e-01]	 2.2226954e-01


.. parsed-literal::

       9	-4.4370024e-03	 2.3860911e-01	 1.4029455e-02	 2.6375158e-01	[-9.2391066e-02]	 2.1055889e-01


.. parsed-literal::

      10	 5.3376174e-03	 2.3689613e-01	 2.1624339e-02	 2.6250360e-01	 -9.7134713e-02 	 2.0849156e-01


.. parsed-literal::

      11	 1.1737512e-02	 2.3583697e-01	 2.7021487e-02	 2.6105951e-01	[-8.7560400e-02]	 2.1339130e-01


.. parsed-literal::

      12	 1.6355856e-02	 2.3501465e-01	 3.1135098e-02	 2.5953602e-01	[-7.6576025e-02]	 2.1519136e-01


.. parsed-literal::

      13	 2.1860654e-02	 2.3400465e-01	 3.6609435e-02	 2.5765088e-01	[-6.7255358e-02]	 2.1517777e-01


.. parsed-literal::

      14	 1.1100128e-01	 2.2186145e-01	 1.3049121e-01	 2.4328233e-01	[ 5.1510249e-02]	 3.3812094e-01


.. parsed-literal::

      15	 1.4170219e-01	 2.1899557e-01	 1.6416153e-01	 2.4105133e-01	[ 1.0521094e-01]	 3.3877802e-01


.. parsed-literal::

      16	 1.9559325e-01	 2.1391438e-01	 2.1916196e-01	 2.3482564e-01	[ 1.6232350e-01]	 2.1427655e-01


.. parsed-literal::

      17	 2.7996742e-01	 2.1035047e-01	 3.0968890e-01	 2.2934150e-01	[ 2.3856994e-01]	 2.2422957e-01


.. parsed-literal::

      18	 3.3336508e-01	 2.0457754e-01	 3.6798411e-01	 2.2529315e-01	[ 2.8219945e-01]	 2.0818996e-01


.. parsed-literal::

      19	 4.0767214e-01	 2.0222042e-01	 4.4390979e-01	 2.2893595e-01	[ 3.3761052e-01]	 2.1829438e-01


.. parsed-literal::

      20	 4.7574367e-01	 2.0018353e-01	 5.1183557e-01	 2.2228811e-01	[ 4.3537896e-01]	 2.1529508e-01


.. parsed-literal::

      21	 5.2336720e-01	 1.9664286e-01	 5.6004743e-01	 2.1829369e-01	[ 4.9361679e-01]	 2.1498275e-01


.. parsed-literal::

      22	 6.4832100e-01	 1.9194758e-01	 6.8836709e-01	 2.0851996e-01	[ 6.3883852e-01]	 2.1171379e-01
      23	 6.8280200e-01	 1.9101240e-01	 7.2431112e-01	 2.0735876e-01	[ 6.8583153e-01]	 2.0722175e-01


.. parsed-literal::

      24	 7.1404139e-01	 1.8684182e-01	 7.5422819e-01	 2.0485575e-01	[ 7.1492108e-01]	 2.0756865e-01


.. parsed-literal::

      25	 7.3775553e-01	 1.8493092e-01	 7.7761375e-01	 2.0277613e-01	[ 7.3808238e-01]	 2.2132564e-01


.. parsed-literal::

      26	 7.5379557e-01	 1.8356409e-01	 7.9322632e-01	 2.0112422e-01	[ 7.5043590e-01]	 3.4471107e-01
      27	 7.7681328e-01	 1.8212520e-01	 8.1631181e-01	 1.9827894e-01	[ 7.6841984e-01]	 1.8489242e-01


.. parsed-literal::

      28	 8.0336900e-01	 1.8046089e-01	 8.4338234e-01	 1.9558533e-01	[ 7.9162420e-01]	 2.0861125e-01


.. parsed-literal::

      29	 8.2711127e-01	 1.8002759e-01	 8.6803990e-01	 1.9582020e-01	[ 8.2015705e-01]	 2.0811296e-01
      30	 8.5891665e-01	 1.7936008e-01	 9.0047202e-01	 1.9620050e-01	[ 8.5006147e-01]	 2.0116138e-01


.. parsed-literal::

      31	 8.9787472e-01	 1.7480762e-01	 9.4191198e-01	 1.9417742e-01	[ 8.8963186e-01]	 2.0492911e-01


.. parsed-literal::

      32	 9.1292485e-01	 1.7435129e-01	 9.5828613e-01	 1.9025952e-01	[ 9.0621726e-01]	 2.1300602e-01


.. parsed-literal::

      33	 9.4335531e-01	 1.6735655e-01	 9.8827634e-01	 1.8379002e-01	[ 9.2941736e-01]	 2.1691799e-01


.. parsed-literal::

      34	 9.5487051e-01	 1.6410431e-01	 9.9979483e-01	 1.8030716e-01	[ 9.4151311e-01]	 2.3398995e-01


.. parsed-literal::

      35	 9.7400986e-01	 1.5963555e-01	 1.0190625e+00	 1.7472289e-01	[ 9.6135192e-01]	 2.1761155e-01
      36	 9.8995084e-01	 1.5791369e-01	 1.0352968e+00	 1.7156437e-01	[ 9.8417367e-01]	 1.9477153e-01


.. parsed-literal::

      37	 1.0070171e+00	 1.5588386e-01	 1.0525336e+00	 1.6801615e-01	[ 1.0025798e+00]	 2.0220017e-01
      38	 1.0283557e+00	 1.5274865e-01	 1.0744546e+00	 1.6407186e-01	[ 1.0248998e+00]	 1.9121552e-01


.. parsed-literal::

      39	 1.0522875e+00	 1.4879520e-01	 1.1000679e+00	 1.6088700e-01	[ 1.0395790e+00]	 1.7648268e-01


.. parsed-literal::

      40	 1.0675563e+00	 1.4595486e-01	 1.1163835e+00	 1.5937854e-01	[ 1.0565806e+00]	 2.0217395e-01
      41	 1.0829969e+00	 1.4348695e-01	 1.1317423e+00	 1.5648129e-01	[ 1.0629680e+00]	 1.9918418e-01


.. parsed-literal::

      42	 1.0919711e+00	 1.4187558e-01	 1.1409936e+00	 1.5458753e-01	[ 1.0646517e+00]	 2.1063542e-01
      43	 1.1056057e+00	 1.3992171e-01	 1.1550031e+00	 1.5232478e-01	[ 1.0702377e+00]	 2.0616913e-01


.. parsed-literal::

      44	 1.1222940e+00	 1.3832497e-01	 1.1726182e+00	 1.5045988e-01	  1.0601912e+00 	 2.1344042e-01


.. parsed-literal::

      45	 1.1290044e+00	 1.3860830e-01	 1.1798687e+00	 1.5065308e-01	  1.0698859e+00 	 2.1702409e-01
      46	 1.1425073e+00	 1.3773088e-01	 1.1925622e+00	 1.4993081e-01	[ 1.0832433e+00]	 2.0357037e-01


.. parsed-literal::

      47	 1.1500121e+00	 1.3770189e-01	 1.1999363e+00	 1.5020686e-01	[ 1.0892989e+00]	 2.0826960e-01
      48	 1.1608811e+00	 1.3823513e-01	 1.2107387e+00	 1.5118260e-01	[ 1.1058927e+00]	 2.0027208e-01


.. parsed-literal::

      49	 1.1679045e+00	 1.3870588e-01	 1.2181513e+00	 1.5291894e-01	[ 1.1146831e+00]	 2.2378039e-01


.. parsed-literal::

      50	 1.1778552e+00	 1.3868652e-01	 1.2279829e+00	 1.5258995e-01	[ 1.1335707e+00]	 2.1226454e-01


.. parsed-literal::

      51	 1.1841699e+00	 1.3770097e-01	 1.2343420e+00	 1.5107411e-01	[ 1.1425062e+00]	 2.1268678e-01


.. parsed-literal::

      52	 1.1944063e+00	 1.3621922e-01	 1.2449297e+00	 1.4906753e-01	[ 1.1533774e+00]	 2.1431136e-01


.. parsed-literal::

      53	 1.2088706e+00	 1.3403995e-01	 1.2599190e+00	 1.4657272e-01	[ 1.1636763e+00]	 2.1794701e-01


.. parsed-literal::

      54	 1.2170643e+00	 1.3287931e-01	 1.2686202e+00	 1.4569661e-01	[ 1.1639629e+00]	 2.1215129e-01


.. parsed-literal::

      55	 1.2248303e+00	 1.3227936e-01	 1.2760755e+00	 1.4518521e-01	[ 1.1688769e+00]	 2.1708465e-01


.. parsed-literal::

      56	 1.2321805e+00	 1.3187787e-01	 1.2835454e+00	 1.4480483e-01	[ 1.1691016e+00]	 2.0642710e-01


.. parsed-literal::

      57	 1.2417831e+00	 1.3105043e-01	 1.2935575e+00	 1.4383658e-01	  1.1670818e+00 	 2.1128225e-01


.. parsed-literal::

      58	 1.2483175e+00	 1.3092978e-01	 1.3007790e+00	 1.4327172e-01	  1.1508731e+00 	 2.1216321e-01


.. parsed-literal::

      59	 1.2600890e+00	 1.2979139e-01	 1.3124229e+00	 1.4225988e-01	  1.1645231e+00 	 2.0766544e-01


.. parsed-literal::

      60	 1.2663266e+00	 1.2935943e-01	 1.3186300e+00	 1.4186748e-01	[ 1.1733595e+00]	 2.0846391e-01


.. parsed-literal::

      61	 1.2747920e+00	 1.2874294e-01	 1.3276151e+00	 1.4142591e-01	  1.1700844e+00 	 2.1638179e-01


.. parsed-literal::

      62	 1.2797036e+00	 1.2863100e-01	 1.3332719e+00	 1.4149545e-01	  1.1674030e+00 	 2.3989987e-01


.. parsed-literal::

      63	 1.2878043e+00	 1.2828798e-01	 1.3411689e+00	 1.4141805e-01	  1.1682570e+00 	 2.1017528e-01
      64	 1.2924592e+00	 1.2817264e-01	 1.3456817e+00	 1.4148909e-01	  1.1693987e+00 	 1.9588614e-01


.. parsed-literal::

      65	 1.2986452e+00	 1.2796954e-01	 1.3518425e+00	 1.4151562e-01	  1.1710617e+00 	 2.1544886e-01


.. parsed-literal::

      66	 1.3071836e+00	 1.2773824e-01	 1.3606224e+00	 1.4151884e-01	[ 1.1762633e+00]	 2.1129632e-01


.. parsed-literal::

      67	 1.3149859e+00	 1.2732824e-01	 1.3687230e+00	 1.4109122e-01	[ 1.1778594e+00]	 2.0872045e-01
      68	 1.3206546e+00	 1.2716651e-01	 1.3743680e+00	 1.4064970e-01	[ 1.1858496e+00]	 2.0347285e-01


.. parsed-literal::

      69	 1.3289573e+00	 1.2699281e-01	 1.3829581e+00	 1.4004979e-01	[ 1.1915158e+00]	 1.9630623e-01
      70	 1.3334998e+00	 1.2671734e-01	 1.3877538e+00	 1.3956645e-01	[ 1.1928432e+00]	 2.0006537e-01


.. parsed-literal::

      71	 1.3398166e+00	 1.2645693e-01	 1.3938392e+00	 1.3933486e-01	[ 1.1985311e+00]	 1.9941735e-01


.. parsed-literal::

      72	 1.3459051e+00	 1.2610632e-01	 1.3999658e+00	 1.3926281e-01	  1.1958399e+00 	 2.1344447e-01


.. parsed-literal::

      73	 1.3503563e+00	 1.2576155e-01	 1.4043924e+00	 1.3900242e-01	  1.1977466e+00 	 2.0633388e-01


.. parsed-literal::

      74	 1.3571936e+00	 1.2532422e-01	 1.4114013e+00	 1.3888235e-01	  1.1962451e+00 	 2.0763540e-01


.. parsed-literal::

      75	 1.3626076e+00	 1.2490556e-01	 1.4170031e+00	 1.3853247e-01	  1.1973206e+00 	 2.1410823e-01
      76	 1.3671654e+00	 1.2459573e-01	 1.4216285e+00	 1.3821024e-01	[ 1.2018067e+00]	 2.0222545e-01


.. parsed-literal::

      77	 1.3720899e+00	 1.2428396e-01	 1.4268120e+00	 1.3782366e-01	  1.1908874e+00 	 2.0896864e-01


.. parsed-literal::

      78	 1.3766793e+00	 1.2413739e-01	 1.4313878e+00	 1.3780755e-01	  1.1896782e+00 	 2.1907520e-01
      79	 1.3816045e+00	 1.2408256e-01	 1.4362890e+00	 1.3802082e-01	  1.1831491e+00 	 1.9371390e-01


.. parsed-literal::

      80	 1.3884037e+00	 1.2404299e-01	 1.4432711e+00	 1.3830166e-01	  1.1621822e+00 	 1.9671369e-01


.. parsed-literal::

      81	 1.3895955e+00	 1.2355597e-01	 1.4448319e+00	 1.3814211e-01	  1.1311106e+00 	 2.1716809e-01


.. parsed-literal::

      82	 1.3949790e+00	 1.2344777e-01	 1.4499781e+00	 1.3807223e-01	  1.1486065e+00 	 2.1057653e-01


.. parsed-literal::

      83	 1.3966003e+00	 1.2328847e-01	 1.4516956e+00	 1.3806091e-01	  1.1482939e+00 	 2.0547581e-01


.. parsed-literal::

      84	 1.3993384e+00	 1.2290518e-01	 1.4545934e+00	 1.3796804e-01	  1.1437938e+00 	 2.1469164e-01


.. parsed-literal::

      85	 1.4033611e+00	 1.2259009e-01	 1.4589048e+00	 1.3851471e-01	  1.1274199e+00 	 2.1636033e-01


.. parsed-literal::

      86	 1.4067471e+00	 1.2239972e-01	 1.4624070e+00	 1.3880281e-01	  1.1149013e+00 	 2.0882821e-01


.. parsed-literal::

      87	 1.4090940e+00	 1.2247960e-01	 1.4647220e+00	 1.3899475e-01	  1.1114235e+00 	 2.1048999e-01
      88	 1.4119956e+00	 1.2253788e-01	 1.4674999e+00	 1.3909423e-01	  1.1152203e+00 	 1.9837904e-01


.. parsed-literal::

      89	 1.4155654e+00	 1.2272351e-01	 1.4711114e+00	 1.3988790e-01	  1.1125416e+00 	 2.1162677e-01


.. parsed-literal::

      90	 1.4188003e+00	 1.2248832e-01	 1.4742820e+00	 1.3965133e-01	  1.1261945e+00 	 2.0595384e-01


.. parsed-literal::

      91	 1.4215873e+00	 1.2226439e-01	 1.4771230e+00	 1.3973918e-01	  1.1269073e+00 	 2.0962405e-01


.. parsed-literal::

      92	 1.4244580e+00	 1.2197411e-01	 1.4801603e+00	 1.3988380e-01	  1.1244691e+00 	 2.1671939e-01


.. parsed-literal::

      93	 1.4284317e+00	 1.2160057e-01	 1.4842595e+00	 1.3967534e-01	  1.1229618e+00 	 2.0886326e-01
      94	 1.4320547e+00	 1.2139010e-01	 1.4880552e+00	 1.3993358e-01	  1.1248800e+00 	 2.0680666e-01


.. parsed-literal::

      95	 1.4350057e+00	 1.2130447e-01	 1.4908228e+00	 1.3956797e-01	  1.1329141e+00 	 2.0199513e-01
      96	 1.4370736e+00	 1.2112647e-01	 1.4927672e+00	 1.3926695e-01	  1.1354311e+00 	 1.9941592e-01


.. parsed-literal::

      97	 1.4388783e+00	 1.2109753e-01	 1.4945328e+00	 1.3934513e-01	  1.1404897e+00 	 2.0276976e-01
      98	 1.4412220e+00	 1.2092461e-01	 1.4969099e+00	 1.3936768e-01	  1.1411394e+00 	 1.9867659e-01


.. parsed-literal::

      99	 1.4447628e+00	 1.2069343e-01	 1.5006232e+00	 1.3946923e-01	  1.1447475e+00 	 2.0647216e-01
     100	 1.4474600e+00	 1.2047329e-01	 1.5035951e+00	 1.3956672e-01	  1.1473450e+00 	 1.9746161e-01


.. parsed-literal::

     101	 1.4506315e+00	 1.2034644e-01	 1.5067651e+00	 1.3919723e-01	  1.1530360e+00 	 2.0215940e-01


.. parsed-literal::

     102	 1.4528745e+00	 1.2026941e-01	 1.5090128e+00	 1.3890611e-01	  1.1586042e+00 	 2.2434020e-01


.. parsed-literal::

     103	 1.4549941e+00	 1.2011577e-01	 1.5111714e+00	 1.3857998e-01	  1.1619790e+00 	 2.2030687e-01


.. parsed-literal::

     104	 1.4576293e+00	 1.1995392e-01	 1.5138288e+00	 1.3834582e-01	  1.1675736e+00 	 2.1358633e-01


.. parsed-literal::

     105	 1.4601025e+00	 1.1971841e-01	 1.5162972e+00	 1.3837381e-01	  1.1656531e+00 	 2.1479201e-01


.. parsed-literal::

     106	 1.4625424e+00	 1.1946302e-01	 1.5187793e+00	 1.3852752e-01	  1.1628702e+00 	 2.2146559e-01


.. parsed-literal::

     107	 1.4638833e+00	 1.1934283e-01	 1.5202736e+00	 1.3880930e-01	  1.1478772e+00 	 2.1375275e-01


.. parsed-literal::

     108	 1.4654620e+00	 1.1926253e-01	 1.5218714e+00	 1.3883838e-01	  1.1441906e+00 	 2.0785856e-01


.. parsed-literal::

     109	 1.4669488e+00	 1.1921425e-01	 1.5234870e+00	 1.3889183e-01	  1.1384348e+00 	 2.1961856e-01


.. parsed-literal::

     110	 1.4682115e+00	 1.1915300e-01	 1.5248715e+00	 1.3888055e-01	  1.1312646e+00 	 2.0827079e-01


.. parsed-literal::

     111	 1.4697626e+00	 1.1909471e-01	 1.5266168e+00	 1.3880527e-01	  1.1206743e+00 	 2.2285533e-01


.. parsed-literal::

     112	 1.4711746e+00	 1.1901962e-01	 1.5279719e+00	 1.3861675e-01	  1.1194843e+00 	 2.1312356e-01


.. parsed-literal::

     113	 1.4722993e+00	 1.1896657e-01	 1.5290060e+00	 1.3843423e-01	  1.1203546e+00 	 2.0585084e-01


.. parsed-literal::

     114	 1.4739826e+00	 1.1890038e-01	 1.5306346e+00	 1.3827770e-01	  1.1187805e+00 	 2.1017075e-01
     115	 1.4769089e+00	 1.1884984e-01	 1.5335991e+00	 1.3819930e-01	  1.1164849e+00 	 1.9107008e-01


.. parsed-literal::

     116	 1.4786911e+00	 1.1885850e-01	 1.5354222e+00	 1.3820976e-01	  1.1138463e+00 	 3.3222723e-01


.. parsed-literal::

     117	 1.4806618e+00	 1.1888591e-01	 1.5374950e+00	 1.3829855e-01	  1.1131385e+00 	 2.1836782e-01
     118	 1.4822413e+00	 1.1895649e-01	 1.5392223e+00	 1.3852973e-01	  1.1121158e+00 	 2.0174646e-01


.. parsed-literal::

     119	 1.4835704e+00	 1.1890445e-01	 1.5406470e+00	 1.3851321e-01	  1.1101415e+00 	 2.0300531e-01


.. parsed-literal::

     120	 1.4853513e+00	 1.1885632e-01	 1.5424914e+00	 1.3849246e-01	  1.1085193e+00 	 2.1274304e-01


.. parsed-literal::

     121	 1.4869959e+00	 1.1872091e-01	 1.5441496e+00	 1.3830687e-01	  1.1121150e+00 	 2.1225476e-01


.. parsed-literal::

     122	 1.4882686e+00	 1.1866668e-01	 1.5453398e+00	 1.3826723e-01	  1.1115856e+00 	 2.2256136e-01


.. parsed-literal::

     123	 1.4899928e+00	 1.1858935e-01	 1.5469768e+00	 1.3824565e-01	  1.1113470e+00 	 2.1315813e-01
     124	 1.4923817e+00	 1.1848975e-01	 1.5493244e+00	 1.3831972e-01	  1.1108199e+00 	 1.9851708e-01


.. parsed-literal::

     125	 1.4939779e+00	 1.1836765e-01	 1.5510037e+00	 1.3831098e-01	  1.1058286e+00 	 3.3583140e-01


.. parsed-literal::

     126	 1.4961301e+00	 1.1828403e-01	 1.5531901e+00	 1.3837541e-01	  1.1045752e+00 	 2.1554399e-01


.. parsed-literal::

     127	 1.4973934e+00	 1.1825783e-01	 1.5545023e+00	 1.3842397e-01	  1.1025274e+00 	 2.1575165e-01


.. parsed-literal::

     128	 1.4989093e+00	 1.1810133e-01	 1.5561183e+00	 1.3814963e-01	  1.0951370e+00 	 2.1263337e-01


.. parsed-literal::

     129	 1.5004001e+00	 1.1797502e-01	 1.5577180e+00	 1.3801567e-01	  1.0878736e+00 	 2.1419978e-01
     130	 1.5017305e+00	 1.1782626e-01	 1.5590449e+00	 1.3780761e-01	  1.0833633e+00 	 2.0721388e-01


.. parsed-literal::

     131	 1.5029768e+00	 1.1762010e-01	 1.5602913e+00	 1.3744732e-01	  1.0784236e+00 	 2.1070361e-01
     132	 1.5040181e+00	 1.1749418e-01	 1.5613401e+00	 1.3735838e-01	  1.0770388e+00 	 1.9682264e-01


.. parsed-literal::

     133	 1.5052930e+00	 1.1732796e-01	 1.5626075e+00	 1.3718573e-01	  1.0767292e+00 	 2.0791698e-01


.. parsed-literal::

     134	 1.5065752e+00	 1.1713625e-01	 1.5639268e+00	 1.3711989e-01	  1.0756754e+00 	 2.1227074e-01


.. parsed-literal::

     135	 1.5075297e+00	 1.1700875e-01	 1.5648893e+00	 1.3699881e-01	  1.0789367e+00 	 2.1341968e-01


.. parsed-literal::

     136	 1.5083494e+00	 1.1696727e-01	 1.5657070e+00	 1.3696965e-01	  1.0792483e+00 	 2.1211147e-01
     137	 1.5099668e+00	 1.1688804e-01	 1.5673672e+00	 1.3684199e-01	  1.0775178e+00 	 1.9248724e-01


.. parsed-literal::

     138	 1.5113455e+00	 1.1685488e-01	 1.5687980e+00	 1.3678625e-01	  1.0752635e+00 	 2.1981215e-01


.. parsed-literal::

     139	 1.5126249e+00	 1.1684052e-01	 1.5702543e+00	 1.3677224e-01	  1.0679518e+00 	 2.0988750e-01


.. parsed-literal::

     140	 1.5141700e+00	 1.1682336e-01	 1.5718094e+00	 1.3679075e-01	  1.0657086e+00 	 2.1480107e-01


.. parsed-literal::

     141	 1.5148580e+00	 1.1681833e-01	 1.5724550e+00	 1.3682230e-01	  1.0661552e+00 	 2.1083093e-01


.. parsed-literal::

     142	 1.5166077e+00	 1.1680929e-01	 1.5741809e+00	 1.3691022e-01	  1.0609290e+00 	 2.0759559e-01


.. parsed-literal::

     143	 1.5180068e+00	 1.1686248e-01	 1.5756698e+00	 1.3725899e-01	  1.0450483e+00 	 2.1285796e-01


.. parsed-literal::

     144	 1.5196929e+00	 1.1687097e-01	 1.5773111e+00	 1.3724511e-01	  1.0407572e+00 	 2.0579505e-01


.. parsed-literal::

     145	 1.5205669e+00	 1.1684598e-01	 1.5781806e+00	 1.3717185e-01	  1.0421266e+00 	 2.0277667e-01


.. parsed-literal::

     146	 1.5219966e+00	 1.1680837e-01	 1.5796611e+00	 1.3713720e-01	  1.0392118e+00 	 2.1084118e-01


.. parsed-literal::

     147	 1.5227274e+00	 1.1693309e-01	 1.5804997e+00	 1.3737075e-01	  1.0366875e+00 	 2.1224689e-01


.. parsed-literal::

     148	 1.5243130e+00	 1.1686414e-01	 1.5820690e+00	 1.3728360e-01	  1.0325633e+00 	 2.0934486e-01
     149	 1.5253989e+00	 1.1685397e-01	 1.5831450e+00	 1.3731471e-01	  1.0270823e+00 	 2.0097327e-01


.. parsed-literal::

     150	 1.5263284e+00	 1.1688343e-01	 1.5840889e+00	 1.3743971e-01	  1.0191371e+00 	 2.1467948e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 9s, sys: 1.14 s, total: 2min 10s
    Wall time: 32.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4b7803f880>



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

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 950 ms, sys: 44.9 ms, total: 995 ms
    Wall time: 377 ms


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




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

