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
       1	-3.5004184e-01	 3.2234701e-01	-3.4042096e-01	 3.1359990e-01	[-3.2454546e-01]	 4.7846484e-01


.. parsed-literal::

       2	-2.7833329e-01	 3.1171990e-01	-2.5462793e-01	 3.0314735e-01	[-2.2949555e-01]	 2.3499560e-01


.. parsed-literal::

       3	-2.3422206e-01	 2.9113322e-01	-1.9238570e-01	 2.8173389e-01	[-1.5543143e-01]	 2.9465365e-01
       4	-1.9773871e-01	 2.6738950e-01	-1.5674469e-01	 2.5740100e-01	[-1.0021416e-01]	 1.9875073e-01


.. parsed-literal::

       5	-1.0960744e-01	 2.5859372e-01	-7.5942214e-02	 2.4915977e-01	[-3.3154532e-02]	 2.1241283e-01
       6	-7.6276653e-02	 2.5327128e-01	-4.6367173e-02	 2.4528078e-01	[-1.6840846e-02]	 1.8373466e-01


.. parsed-literal::

       7	-5.8680861e-02	 2.5051099e-01	-3.4963676e-02	 2.4206177e-01	[-2.6094683e-03]	 2.0740533e-01


.. parsed-literal::

       8	-4.6172279e-02	 2.4839963e-01	-2.6213681e-02	 2.3984681e-01	[ 7.2120884e-03]	 2.0744681e-01


.. parsed-literal::

       9	-3.2910933e-02	 2.4591763e-01	-1.5719857e-02	 2.3731867e-01	[ 1.8254431e-02]	 2.0980835e-01


.. parsed-literal::

      10	-2.2630255e-02	 2.4403091e-01	-7.5347751e-03	 2.3509462e-01	[ 2.9417682e-02]	 2.1862745e-01
      11	-1.8016048e-02	 2.4330210e-01	-3.9605514e-03	 2.3460751e-01	[ 3.0390944e-02]	 1.9829822e-01


.. parsed-literal::

      12	-1.5517321e-02	 2.4278002e-01	-1.6213405e-03	 2.3438224e-01	[ 3.2890655e-02]	 2.1254611e-01


.. parsed-literal::

      13	-1.1653478e-02	 2.4199523e-01	 2.2044706e-03	 2.3393339e-01	[ 3.5617537e-02]	 2.0502353e-01


.. parsed-literal::

      14	-6.2426999e-03	 2.4072997e-01	 8.6611349e-03	 2.3291100e-01	[ 4.3170810e-02]	 2.1054959e-01


.. parsed-literal::

      15	 1.0801705e-01	 2.2605453e-01	 1.3146597e-01	 2.1882086e-01	[ 1.7665316e-01]	 2.1200609e-01


.. parsed-literal::

      16	 1.7830032e-01	 2.2387629e-01	 2.0613523e-01	 2.1724606e-01	[ 2.2013009e-01]	 3.2391524e-01


.. parsed-literal::

      17	 2.8550842e-01	 2.1442864e-01	 3.1605943e-01	 2.0562926e-01	[ 3.3877069e-01]	 2.1817756e-01


.. parsed-literal::

      18	 3.3199915e-01	 2.1049481e-01	 3.6479034e-01	 2.0194306e-01	[ 3.8931282e-01]	 3.2451224e-01


.. parsed-literal::

      19	 3.7100783e-01	 2.0618213e-01	 4.0540325e-01	 1.9867774e-01	[ 4.2357228e-01]	 2.0595264e-01


.. parsed-literal::

      20	 4.2622586e-01	 2.0268767e-01	 4.6099659e-01	 1.9637618e-01	[ 4.6571434e-01]	 2.0385599e-01
      21	 5.3644212e-01	 2.0002853e-01	 5.7375155e-01	 1.9235368e-01	[ 5.4476547e-01]	 1.7796874e-01


.. parsed-literal::

      22	 6.0710989e-01	 1.9594268e-01	 6.4706756e-01	 1.8511079e-01	[ 5.9030252e-01]	 2.0940185e-01


.. parsed-literal::

      23	 6.3929036e-01	 1.9412133e-01	 6.7607829e-01	 1.8330400e-01	[ 6.4106105e-01]	 2.0983982e-01


.. parsed-literal::

      24	 6.7494726e-01	 1.9249279e-01	 7.1332292e-01	 1.8164242e-01	[ 6.7388034e-01]	 2.0377374e-01


.. parsed-literal::

      25	 7.1385355e-01	 1.8984405e-01	 7.5328793e-01	 1.7985332e-01	[ 7.0822927e-01]	 2.1744704e-01
      26	 7.6119631e-01	 1.9027247e-01	 8.0068893e-01	 1.8098574e-01	[ 7.5374048e-01]	 1.8924236e-01


.. parsed-literal::

      27	 8.0222468e-01	 1.8857500e-01	 8.4125981e-01	 1.7973712e-01	[ 8.0156891e-01]	 1.9964385e-01


.. parsed-literal::

      28	 8.3016997e-01	 1.8799264e-01	 8.6956399e-01	 1.7891277e-01	[ 8.3557586e-01]	 2.1121693e-01


.. parsed-literal::

      29	 8.5783167e-01	 1.9125836e-01	 8.9803984e-01	 1.8112797e-01	[ 8.7378479e-01]	 2.2029471e-01
      30	 8.8320359e-01	 1.9103530e-01	 9.2494010e-01	 1.8035963e-01	[ 8.9676442e-01]	 1.8816829e-01


.. parsed-literal::

      31	 9.0308768e-01	 1.8765685e-01	 9.4534263e-01	 1.7729933e-01	[ 9.1333736e-01]	 2.0427489e-01


.. parsed-literal::

      32	 9.2033619e-01	 1.8421114e-01	 9.6265359e-01	 1.7479500e-01	[ 9.2233464e-01]	 2.2189736e-01


.. parsed-literal::

      33	 9.3853988e-01	 1.8173152e-01	 9.8079902e-01	 1.7287272e-01	[ 9.3877562e-01]	 2.1246982e-01


.. parsed-literal::

      34	 9.5507247e-01	 1.7997562e-01	 9.9727481e-01	 1.7107849e-01	[ 9.5553856e-01]	 2.0923805e-01


.. parsed-literal::

      35	 9.6479805e-01	 1.7759196e-01	 1.0072758e+00	 1.6891754e-01	[ 9.5743092e-01]	 2.0558476e-01


.. parsed-literal::

      36	 9.8014675e-01	 1.7586419e-01	 1.0225853e+00	 1.6757512e-01	[ 9.7607369e-01]	 2.1556950e-01
      37	 9.9601567e-01	 1.7399268e-01	 1.0390195e+00	 1.6588769e-01	[ 9.9281430e-01]	 1.9423270e-01


.. parsed-literal::

      38	 1.0183090e+00	 1.6999133e-01	 1.0626613e+00	 1.6166949e-01	[ 1.0113414e+00]	 2.0986319e-01


.. parsed-literal::

      39	 1.0339782e+00	 1.6563282e-01	 1.0796279e+00	 1.5599326e-01	[ 1.0319147e+00]	 2.1044159e-01


.. parsed-literal::

      40	 1.0494925e+00	 1.6387180e-01	 1.0951154e+00	 1.5390098e-01	[ 1.0475524e+00]	 2.0407057e-01


.. parsed-literal::

      41	 1.0617940e+00	 1.6158420e-01	 1.1073780e+00	 1.5113059e-01	[ 1.0608343e+00]	 2.1059537e-01


.. parsed-literal::

      42	 1.0781968e+00	 1.5803550e-01	 1.1242808e+00	 1.4688900e-01	[ 1.0775217e+00]	 2.0547414e-01
      43	 1.0923573e+00	 1.5518534e-01	 1.1390376e+00	 1.4273671e-01	[ 1.0990982e+00]	 1.9588733e-01


.. parsed-literal::

      44	 1.1055210e+00	 1.5299075e-01	 1.1522389e+00	 1.4069156e-01	[ 1.1119009e+00]	 2.1116972e-01
      45	 1.1146437e+00	 1.5275546e-01	 1.1610660e+00	 1.4106992e-01	[ 1.1200919e+00]	 2.0359325e-01


.. parsed-literal::

      46	 1.1244129e+00	 1.5123593e-01	 1.1708505e+00	 1.3980254e-01	[ 1.1285795e+00]	 2.1249056e-01


.. parsed-literal::

      47	 1.1330985e+00	 1.5082320e-01	 1.1794988e+00	 1.3945682e-01	[ 1.1365199e+00]	 2.0542645e-01


.. parsed-literal::

      48	 1.1421802e+00	 1.4931492e-01	 1.1886922e+00	 1.3801596e-01	[ 1.1420623e+00]	 2.1533823e-01
      49	 1.1598636e+00	 1.4584028e-01	 1.2071574e+00	 1.3581555e-01	[ 1.1506473e+00]	 1.8289828e-01


.. parsed-literal::

      50	 1.1679947e+00	 1.4534182e-01	 1.2153367e+00	 1.3535899e-01	  1.1499154e+00 	 3.3909988e-01


.. parsed-literal::

      51	 1.1753003e+00	 1.4463203e-01	 1.2225774e+00	 1.3470044e-01	[ 1.1574090e+00]	 2.1721840e-01


.. parsed-literal::

      52	 1.1874812e+00	 1.4351441e-01	 1.2350134e+00	 1.3406622e-01	[ 1.1674669e+00]	 2.2006488e-01


.. parsed-literal::

      53	 1.1968822e+00	 1.4237147e-01	 1.2446587e+00	 1.3350238e-01	[ 1.1695280e+00]	 2.1010709e-01
      54	 1.2047025e+00	 1.4174330e-01	 1.2526709e+00	 1.3300691e-01	[ 1.1728379e+00]	 2.0533419e-01


.. parsed-literal::

      55	 1.2142871e+00	 1.4093273e-01	 1.2624637e+00	 1.3162120e-01	[ 1.1798383e+00]	 1.9439650e-01


.. parsed-literal::

      56	 1.2236135e+00	 1.4046776e-01	 1.2719987e+00	 1.3053664e-01	[ 1.1886428e+00]	 2.2146058e-01
      57	 1.2323058e+00	 1.3962095e-01	 1.2807134e+00	 1.2941557e-01	[ 1.1983794e+00]	 1.8805218e-01


.. parsed-literal::

      58	 1.2428874e+00	 1.3819557e-01	 1.2915768e+00	 1.2756115e-01	[ 1.2084882e+00]	 2.1048045e-01


.. parsed-literal::

      59	 1.2515176e+00	 1.3708179e-01	 1.3004662e+00	 1.2677603e-01	[ 1.2130385e+00]	 2.2043014e-01


.. parsed-literal::

      60	 1.2602143e+00	 1.3640953e-01	 1.3090859e+00	 1.2584637e-01	[ 1.2212740e+00]	 2.1123409e-01


.. parsed-literal::

      61	 1.2700898e+00	 1.3579045e-01	 1.3189785e+00	 1.2477150e-01	[ 1.2308387e+00]	 2.1457505e-01


.. parsed-literal::

      62	 1.2778300e+00	 1.3559804e-01	 1.3267334e+00	 1.2421970e-01	[ 1.2355249e+00]	 2.2489166e-01


.. parsed-literal::

      63	 1.2876232e+00	 1.3533875e-01	 1.3368674e+00	 1.2286812e-01	[ 1.2459845e+00]	 2.1841669e-01
      64	 1.2953358e+00	 1.3552790e-01	 1.3445761e+00	 1.2306182e-01	[ 1.2525544e+00]	 1.9799900e-01


.. parsed-literal::

      65	 1.3006077e+00	 1.3530766e-01	 1.3498254e+00	 1.2303679e-01	[ 1.2585813e+00]	 2.1714306e-01


.. parsed-literal::

      66	 1.3097253e+00	 1.3487206e-01	 1.3591544e+00	 1.2305302e-01	[ 1.2676112e+00]	 2.1034598e-01


.. parsed-literal::

      67	 1.3181462e+00	 1.3442147e-01	 1.3677863e+00	 1.2312704e-01	[ 1.2802938e+00]	 2.1363306e-01


.. parsed-literal::

      68	 1.3266721e+00	 1.3430628e-01	 1.3763688e+00	 1.2320850e-01	[ 1.2868510e+00]	 2.2065306e-01
      69	 1.3327084e+00	 1.3403937e-01	 1.3824773e+00	 1.2290234e-01	[ 1.2904338e+00]	 1.9205546e-01


.. parsed-literal::

      70	 1.3399748e+00	 1.3354510e-01	 1.3900406e+00	 1.2257629e-01	[ 1.2925953e+00]	 2.0034266e-01
      71	 1.3476289e+00	 1.3297204e-01	 1.3980246e+00	 1.2191904e-01	[ 1.2930510e+00]	 1.8171644e-01


.. parsed-literal::

      72	 1.3560824e+00	 1.3222506e-01	 1.4064935e+00	 1.2151357e-01	[ 1.3034531e+00]	 2.0100141e-01


.. parsed-literal::

      73	 1.3624229e+00	 1.3182246e-01	 1.4126224e+00	 1.2131414e-01	[ 1.3154733e+00]	 2.0254755e-01
      74	 1.3681217e+00	 1.3140838e-01	 1.4184784e+00	 1.2084277e-01	[ 1.3239047e+00]	 2.0357800e-01


.. parsed-literal::

      75	 1.3731681e+00	 1.3090537e-01	 1.4236836e+00	 1.2054288e-01	[ 1.3278623e+00]	 2.1113014e-01


.. parsed-literal::

      76	 1.3783497e+00	 1.3070154e-01	 1.4289828e+00	 1.2022780e-01	[ 1.3308247e+00]	 2.1289921e-01


.. parsed-literal::

      77	 1.3819508e+00	 1.3059205e-01	 1.4327680e+00	 1.2017568e-01	  1.3299992e+00 	 2.0765305e-01


.. parsed-literal::

      78	 1.3865906e+00	 1.3035474e-01	 1.4374932e+00	 1.2005991e-01	  1.3304936e+00 	 2.1661973e-01


.. parsed-literal::

      79	 1.3922481e+00	 1.3015253e-01	 1.4432428e+00	 1.2018636e-01	[ 1.3309354e+00]	 2.0982361e-01


.. parsed-literal::

      80	 1.3977408e+00	 1.2995688e-01	 1.4488051e+00	 1.2035896e-01	[ 1.3339118e+00]	 2.0711231e-01
      81	 1.4040959e+00	 1.2967521e-01	 1.4554141e+00	 1.2022201e-01	[ 1.3362557e+00]	 1.7789841e-01


.. parsed-literal::

      82	 1.4093753e+00	 1.2931677e-01	 1.4606197e+00	 1.1984851e-01	[ 1.3444595e+00]	 2.0034027e-01
      83	 1.4137526e+00	 1.2896895e-01	 1.4650190e+00	 1.1926236e-01	[ 1.3499176e+00]	 1.8531013e-01


.. parsed-literal::

      84	 1.4191917e+00	 1.2811711e-01	 1.4704935e+00	 1.1819580e-01	[ 1.3586671e+00]	 2.2061825e-01


.. parsed-literal::

      85	 1.4224193e+00	 1.2771808e-01	 1.4738181e+00	 1.1756955e-01	[ 1.3616814e+00]	 2.0505118e-01


.. parsed-literal::

      86	 1.4264192e+00	 1.2725236e-01	 1.4777631e+00	 1.1721864e-01	[ 1.3649447e+00]	 2.1361995e-01


.. parsed-literal::

      87	 1.4292711e+00	 1.2690830e-01	 1.4806363e+00	 1.1702953e-01	[ 1.3661350e+00]	 2.0674682e-01


.. parsed-literal::

      88	 1.4320818e+00	 1.2667409e-01	 1.4835231e+00	 1.1691753e-01	  1.3658051e+00 	 2.2168541e-01


.. parsed-literal::

      89	 1.4331179e+00	 1.2641618e-01	 1.4850465e+00	 1.1679746e-01	  1.3637401e+00 	 2.1956277e-01


.. parsed-literal::

      90	 1.4390253e+00	 1.2634835e-01	 1.4907446e+00	 1.1673365e-01	  1.3656371e+00 	 2.1580195e-01


.. parsed-literal::

      91	 1.4406330e+00	 1.2631539e-01	 1.4923705e+00	 1.1673065e-01	[ 1.3670826e+00]	 2.2011709e-01


.. parsed-literal::

      92	 1.4433934e+00	 1.2628537e-01	 1.4952127e+00	 1.1685261e-01	[ 1.3676576e+00]	 2.1512914e-01


.. parsed-literal::

      93	 1.4477629e+00	 1.2628461e-01	 1.4996915e+00	 1.1701311e-01	[ 1.3696820e+00]	 2.1417522e-01


.. parsed-literal::

      94	 1.4505044e+00	 1.2617725e-01	 1.5024540e+00	 1.1703657e-01	[ 1.3717323e+00]	 3.3531427e-01
      95	 1.4538510e+00	 1.2615316e-01	 1.5057864e+00	 1.1688255e-01	[ 1.3749711e+00]	 1.8803835e-01


.. parsed-literal::

      96	 1.4576146e+00	 1.2600231e-01	 1.5095209e+00	 1.1644466e-01	[ 1.3779666e+00]	 2.1761370e-01
      97	 1.4603123e+00	 1.2593152e-01	 1.5123070e+00	 1.1611770e-01	  1.3769504e+00 	 1.9507957e-01


.. parsed-literal::

      98	 1.4630201e+00	 1.2582456e-01	 1.5150351e+00	 1.1600148e-01	  1.3774792e+00 	 2.1715450e-01


.. parsed-literal::

      99	 1.4660013e+00	 1.2579823e-01	 1.5181492e+00	 1.1620550e-01	  1.3757461e+00 	 2.1406412e-01


.. parsed-literal::

     100	 1.4686540e+00	 1.2573597e-01	 1.5208908e+00	 1.1632220e-01	  1.3746809e+00 	 2.1513462e-01


.. parsed-literal::

     101	 1.4714461e+00	 1.2571584e-01	 1.5238096e+00	 1.1642283e-01	  1.3746302e+00 	 2.2222090e-01


.. parsed-literal::

     102	 1.4733989e+00	 1.2547528e-01	 1.5259699e+00	 1.1621047e-01	  1.3732187e+00 	 2.2380519e-01


.. parsed-literal::

     103	 1.4761724e+00	 1.2539881e-01	 1.5286733e+00	 1.1597507e-01	  1.3761318e+00 	 2.0823765e-01


.. parsed-literal::

     104	 1.4782045e+00	 1.2522493e-01	 1.5306626e+00	 1.1562802e-01	[ 1.3780421e+00]	 2.0794296e-01


.. parsed-literal::

     105	 1.4805279e+00	 1.2500138e-01	 1.5329959e+00	 1.1533192e-01	[ 1.3788991e+00]	 2.0822954e-01


.. parsed-literal::

     106	 1.4841387e+00	 1.2458213e-01	 1.5367505e+00	 1.1488967e-01	  1.3768932e+00 	 2.0578527e-01


.. parsed-literal::

     107	 1.4866248e+00	 1.2444985e-01	 1.5393296e+00	 1.1515076e-01	  1.3754538e+00 	 2.1888661e-01


.. parsed-literal::

     108	 1.4886458e+00	 1.2434104e-01	 1.5413002e+00	 1.1495736e-01	  1.3773888e+00 	 2.1822715e-01
     109	 1.4901091e+00	 1.2433826e-01	 1.5427724e+00	 1.1516425e-01	  1.3775221e+00 	 1.9514418e-01


.. parsed-literal::

     110	 1.4919559e+00	 1.2429123e-01	 1.5446494e+00	 1.1531295e-01	  1.3769697e+00 	 2.1088171e-01


.. parsed-literal::

     111	 1.4948276e+00	 1.2414880e-01	 1.5475950e+00	 1.1578409e-01	  1.3753662e+00 	 2.1827722e-01


.. parsed-literal::

     112	 1.4968506e+00	 1.2402111e-01	 1.5496568e+00	 1.1592312e-01	  1.3711836e+00 	 2.2397971e-01


.. parsed-literal::

     113	 1.4984451e+00	 1.2390984e-01	 1.5512217e+00	 1.1589091e-01	  1.3710383e+00 	 2.0479298e-01


.. parsed-literal::

     114	 1.5007825e+00	 1.2370105e-01	 1.5535976e+00	 1.1600324e-01	  1.3674756e+00 	 2.2099519e-01


.. parsed-literal::

     115	 1.5024428e+00	 1.2361878e-01	 1.5553154e+00	 1.1608071e-01	  1.3646400e+00 	 2.1686935e-01


.. parsed-literal::

     116	 1.5044325e+00	 1.2353537e-01	 1.5573229e+00	 1.1618937e-01	  1.3621935e+00 	 2.1616268e-01


.. parsed-literal::

     117	 1.5066812e+00	 1.2342624e-01	 1.5596120e+00	 1.1614209e-01	  1.3584450e+00 	 2.1429777e-01
     118	 1.5082692e+00	 1.2330614e-01	 1.5612101e+00	 1.1604156e-01	  1.3562476e+00 	 1.9622278e-01


.. parsed-literal::

     119	 1.5096110e+00	 1.2317940e-01	 1.5625221e+00	 1.1572269e-01	  1.3566656e+00 	 2.1404505e-01


.. parsed-literal::

     120	 1.5117572e+00	 1.2296500e-01	 1.5646950e+00	 1.1539441e-01	  1.3544680e+00 	 2.2327614e-01
     121	 1.5127844e+00	 1.2280975e-01	 1.5657839e+00	 1.1506169e-01	  1.3522173e+00 	 1.9487572e-01


.. parsed-literal::

     122	 1.5138173e+00	 1.2281125e-01	 1.5668060e+00	 1.1515240e-01	  1.3533148e+00 	 2.1379828e-01
     123	 1.5165315e+00	 1.2282238e-01	 1.5695847e+00	 1.1539617e-01	  1.3560888e+00 	 1.8860674e-01


.. parsed-literal::

     124	 1.5178447e+00	 1.2276058e-01	 1.5708938e+00	 1.1518845e-01	  1.3578819e+00 	 2.0970416e-01
     125	 1.5201507e+00	 1.2268477e-01	 1.5732111e+00	 1.1462026e-01	  1.3596635e+00 	 1.9368768e-01


.. parsed-literal::

     126	 1.5215780e+00	 1.2261405e-01	 1.5746408e+00	 1.1442152e-01	  1.3605958e+00 	 2.1960902e-01


.. parsed-literal::

     127	 1.5228246e+00	 1.2254883e-01	 1.5758562e+00	 1.1433785e-01	  1.3604791e+00 	 2.1840167e-01


.. parsed-literal::

     128	 1.5250292e+00	 1.2238219e-01	 1.5780850e+00	 1.1429546e-01	  1.3578221e+00 	 2.2418404e-01


.. parsed-literal::

     129	 1.5263444e+00	 1.2227739e-01	 1.5794739e+00	 1.1422822e-01	  1.3530121e+00 	 2.1777701e-01
     130	 1.5278097e+00	 1.2215826e-01	 1.5809743e+00	 1.1412352e-01	  1.3516102e+00 	 1.9943523e-01


.. parsed-literal::

     131	 1.5303073e+00	 1.2200910e-01	 1.5835762e+00	 1.1414178e-01	  1.3497807e+00 	 2.1237564e-01


.. parsed-literal::

     132	 1.5314946e+00	 1.2195029e-01	 1.5847892e+00	 1.1404573e-01	  1.3496337e+00 	 2.1516514e-01


.. parsed-literal::

     133	 1.5325075e+00	 1.2195413e-01	 1.5857456e+00	 1.1411518e-01	  1.3510752e+00 	 2.1829677e-01


.. parsed-literal::

     134	 1.5333701e+00	 1.2199396e-01	 1.5865729e+00	 1.1427043e-01	  1.3518185e+00 	 2.1576667e-01
     135	 1.5343221e+00	 1.2198299e-01	 1.5875204e+00	 1.1439063e-01	  1.3509752e+00 	 1.9996476e-01


.. parsed-literal::

     136	 1.5368508e+00	 1.2192148e-01	 1.5900904e+00	 1.1448872e-01	  1.3477447e+00 	 2.0634818e-01


.. parsed-literal::

     137	 1.5376579e+00	 1.2185427e-01	 1.5909454e+00	 1.1449283e-01	  1.3455175e+00 	 3.0193591e-01


.. parsed-literal::

     138	 1.5388126e+00	 1.2176960e-01	 1.5921463e+00	 1.1432642e-01	  1.3426180e+00 	 2.1709371e-01
     139	 1.5398971e+00	 1.2168615e-01	 1.5932773e+00	 1.1413514e-01	  1.3417486e+00 	 1.8474650e-01


.. parsed-literal::

     140	 1.5407257e+00	 1.2153669e-01	 1.5942429e+00	 1.1376460e-01	  1.3394779e+00 	 2.2062683e-01


.. parsed-literal::

     141	 1.5419307e+00	 1.2150895e-01	 1.5954045e+00	 1.1370297e-01	  1.3407475e+00 	 2.1039534e-01
     142	 1.5426916e+00	 1.2149244e-01	 1.5961444e+00	 1.1367384e-01	  1.3423230e+00 	 1.9065213e-01


.. parsed-literal::

     143	 1.5436062e+00	 1.2145132e-01	 1.5970528e+00	 1.1358651e-01	  1.3430427e+00 	 1.8296242e-01


.. parsed-literal::

     144	 1.5447869e+00	 1.2135497e-01	 1.5982538e+00	 1.1326171e-01	  1.3459238e+00 	 2.1294522e-01


.. parsed-literal::

     145	 1.5459054e+00	 1.2131997e-01	 1.5993460e+00	 1.1312455e-01	  1.3459939e+00 	 2.1483135e-01
     146	 1.5467197e+00	 1.2129725e-01	 1.6001425e+00	 1.1306977e-01	  1.3454286e+00 	 1.7492867e-01


.. parsed-literal::

     147	 1.5476154e+00	 1.2127664e-01	 1.6010461e+00	 1.1303610e-01	  1.3439783e+00 	 2.1786976e-01


.. parsed-literal::

     148	 1.5486802e+00	 1.2127160e-01	 1.6021351e+00	 1.1311197e-01	  1.3402832e+00 	 2.0672679e-01


.. parsed-literal::

     149	 1.5499412e+00	 1.2127925e-01	 1.6034555e+00	 1.1323994e-01	  1.3356499e+00 	 2.2356510e-01


.. parsed-literal::

     150	 1.5509198e+00	 1.2129038e-01	 1.6044433e+00	 1.1329868e-01	  1.3332598e+00 	 2.6784253e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 8s, sys: 1.08 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3bf4dfbb50>



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
    CPU times: user 2.21 s, sys: 39 ms, total: 2.25 s
    Wall time: 706 ms


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

