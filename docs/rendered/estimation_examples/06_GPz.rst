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
       1	-3.4387043e-01	 3.2041566e-01	-3.3415884e-01	 3.2135189e-01	[-3.3601907e-01]	 4.6688390e-01


.. parsed-literal::

       2	-2.7081179e-01	 3.0911842e-01	-2.4609427e-01	 3.1023267e-01	[-2.5006216e-01]	 2.2733688e-01


.. parsed-literal::

       3	-2.2619585e-01	 2.8860199e-01	-1.8370866e-01	 2.9152953e-01	[-1.9501965e-01]	 3.8705444e-01
       4	-1.9726105e-01	 2.6501218e-01	-1.5679548e-01	 2.6902125e-01	[-1.8358530e-01]	 1.9623232e-01


.. parsed-literal::

       5	-1.0564441e-01	 2.5726337e-01	-6.9888219e-02	 2.6291195e-01	[-9.6508546e-02]	 2.0349360e-01


.. parsed-literal::

       6	-6.9863044e-02	 2.5104412e-01	-3.7963708e-02	 2.5362933e-01	[-4.8458266e-02]	 2.0575023e-01


.. parsed-literal::

       7	-5.1845704e-02	 2.4844699e-01	-2.6810179e-02	 2.5151358e-01	[-3.9498841e-02]	 2.0577407e-01


.. parsed-literal::

       8	-3.8685538e-02	 2.4630274e-01	-1.7790413e-02	 2.4956850e-01	[-3.1589455e-02]	 2.0998359e-01
       9	-2.4022875e-02	 2.4360182e-01	-6.3064626e-03	 2.4768822e-01	[-2.3429449e-02]	 1.9963765e-01


.. parsed-literal::

      10	-1.3903086e-02	 2.4168872e-01	 1.4372394e-03	 2.4562848e-01	[-1.6468714e-02]	 2.0978689e-01


.. parsed-literal::

      11	-8.1881070e-03	 2.4082718e-01	 6.1921088e-03	 2.4502776e-01	[-1.1243872e-02]	 2.0816302e-01
      12	-5.6337494e-03	 2.4032548e-01	 8.5019578e-03	 2.4460128e-01	[-9.5158295e-03]	 1.7930913e-01


.. parsed-literal::

      13	-1.4345159e-03	 2.3952734e-01	 1.2516689e-02	 2.4369645e-01	[-5.2671148e-03]	 2.0391536e-01


.. parsed-literal::

      14	 4.6572925e-03	 2.3811351e-01	 1.9931234e-02	 2.4204173e-01	[ 2.1971080e-03]	 2.0441842e-01


.. parsed-literal::

      15	 2.1638138e-01	 2.2379247e-01	 2.4437919e-01	 2.2623098e-01	[ 2.3264965e-01]	 3.2209921e-01


.. parsed-literal::

      16	 2.9854145e-01	 2.1746441e-01	 3.2969868e-01	 2.2361676e-01	[ 2.9917922e-01]	 2.0386982e-01


.. parsed-literal::

      17	 3.5679915e-01	 2.1255106e-01	 3.8925450e-01	 2.2135203e-01	[ 3.4269736e-01]	 2.0412779e-01
      18	 4.2683695e-01	 2.0603429e-01	 4.6093514e-01	 2.1265188e-01	[ 4.2182167e-01]	 1.9972992e-01


.. parsed-literal::

      19	 5.2012875e-01	 1.9990089e-01	 5.5624778e-01	 2.0740244e-01	[ 5.1023814e-01]	 2.1275139e-01


.. parsed-literal::

      20	 6.0736481e-01	 1.9855470e-01	 6.4702744e-01	 2.0808651e-01	[ 5.8330888e-01]	 2.0775199e-01


.. parsed-literal::

      21	 6.2710551e-01	 1.9534468e-01	 6.6389941e-01	 2.0593026e-01	[ 6.2418720e-01]	 2.1448374e-01


.. parsed-literal::

      22	 6.6540823e-01	 1.9587877e-01	 7.0397525e-01	 2.0488042e-01	[ 6.6330467e-01]	 2.1364331e-01
      23	 6.9534653e-01	 1.9925606e-01	 7.3452176e-01	 2.0704874e-01	[ 7.0192164e-01]	 1.8359160e-01


.. parsed-literal::

      24	 7.3544049e-01	 1.9641470e-01	 7.7416204e-01	 2.0611121e-01	[ 7.3849439e-01]	 2.0711327e-01
      25	 7.8036537e-01	 1.9853491e-01	 8.1927248e-01	 2.0948511e-01	[ 7.8271236e-01]	 1.8379378e-01


.. parsed-literal::

      26	 8.1334290e-01	 1.9847338e-01	 8.5291335e-01	 2.1181978e-01	[ 8.1989746e-01]	 2.0704818e-01


.. parsed-literal::

      27	 8.4731366e-01	 1.9636563e-01	 8.8800956e-01	 2.0958751e-01	[ 8.4925480e-01]	 2.0356917e-01


.. parsed-literal::

      28	 8.8154679e-01	 1.9248351e-01	 9.2531316e-01	 2.0342469e-01	[ 8.6357411e-01]	 2.1793103e-01
      29	 8.9950048e-01	 1.9176335e-01	 9.4335482e-01	 2.0216607e-01	[ 8.8583896e-01]	 2.0019817e-01


.. parsed-literal::

      30	 9.2029368e-01	 1.8797620e-01	 9.6454949e-01	 1.9693951e-01	[ 9.1198732e-01]	 2.0752835e-01
      31	 9.4884498e-01	 1.8602342e-01	 9.9330609e-01	 1.9247679e-01	[ 9.4584962e-01]	 1.8008733e-01


.. parsed-literal::

      32	 9.7156625e-01	 1.8183005e-01	 1.0172515e+00	 1.8752744e-01	[ 9.7363164e-01]	 2.1132994e-01


.. parsed-literal::

      33	 9.8856387e-01	 1.8074540e-01	 1.0343102e+00	 1.8665539e-01	[ 9.8308270e-01]	 2.2845149e-01
      34	 1.0015799e+00	 1.7955035e-01	 1.0473204e+00	 1.8650312e-01	[ 9.9338306e-01]	 1.9549441e-01


.. parsed-literal::

      35	 1.0193913e+00	 1.7547400e-01	 1.0658796e+00	 1.8485398e-01	[ 1.0084593e+00]	 2.1112037e-01
      36	 1.0342947e+00	 1.7279854e-01	 1.0804106e+00	 1.8416438e-01	[ 1.0290928e+00]	 1.8416214e-01


.. parsed-literal::

      37	 1.0481854e+00	 1.6893125e-01	 1.0944298e+00	 1.8099007e-01	[ 1.0470594e+00]	 2.0758247e-01


.. parsed-literal::

      38	 1.0665174e+00	 1.6145793e-01	 1.1138377e+00	 1.7594898e-01	[ 1.0616033e+00]	 2.1638870e-01


.. parsed-literal::

      39	 1.0781692e+00	 1.5713499e-01	 1.1254889e+00	 1.7177630e-01	[ 1.0768997e+00]	 2.0896983e-01


.. parsed-literal::

      40	 1.0919867e+00	 1.5670405e-01	 1.1388515e+00	 1.7134417e-01	[ 1.0891958e+00]	 2.3255825e-01


.. parsed-literal::

      41	 1.1075580e+00	 1.5475256e-01	 1.1545767e+00	 1.7052069e-01	[ 1.0993797e+00]	 3.1637597e-01


.. parsed-literal::

      42	 1.1215152e+00	 1.5194725e-01	 1.1687397e+00	 1.6900394e-01	[ 1.1069468e+00]	 2.1576905e-01


.. parsed-literal::

      43	 1.1446498e+00	 1.4700948e-01	 1.1928912e+00	 1.6463369e-01	[ 1.1190956e+00]	 2.1605682e-01


.. parsed-literal::

      44	 1.1513102e+00	 1.4604937e-01	 1.1994862e+00	 1.6513297e-01	[ 1.1223203e+00]	 2.1126533e-01


.. parsed-literal::

      45	 1.1644147e+00	 1.4643080e-01	 1.2124563e+00	 1.6442452e-01	[ 1.1405848e+00]	 2.1030474e-01
      46	 1.1738042e+00	 1.4414110e-01	 1.2221117e+00	 1.6078423e-01	[ 1.1491705e+00]	 1.9550347e-01


.. parsed-literal::

      47	 1.1833455e+00	 1.4217753e-01	 1.2320289e+00	 1.5970234e-01	[ 1.1497907e+00]	 2.1113896e-01
      48	 1.1965919e+00	 1.3936414e-01	 1.2461923e+00	 1.5841532e-01	  1.1454579e+00 	 1.9757557e-01


.. parsed-literal::

      49	 1.2045566e+00	 1.3846560e-01	 1.2539876e+00	 1.5927769e-01	  1.1423580e+00 	 1.9882202e-01


.. parsed-literal::

      50	 1.2142312e+00	 1.3786255e-01	 1.2633238e+00	 1.5876535e-01	[ 1.1521922e+00]	 2.1486163e-01


.. parsed-literal::

      51	 1.2253263e+00	 1.3659911e-01	 1.2747209e+00	 1.5802426e-01	[ 1.1526435e+00]	 2.1382737e-01


.. parsed-literal::

      52	 1.2372237e+00	 1.3540899e-01	 1.2868430e+00	 1.5623024e-01	[ 1.1596135e+00]	 2.1324325e-01


.. parsed-literal::

      53	 1.2433987e+00	 1.3446602e-01	 1.2933762e+00	 1.5478326e-01	  1.1496856e+00 	 2.0574689e-01


.. parsed-literal::

      54	 1.2573319e+00	 1.3420659e-01	 1.3073780e+00	 1.5430208e-01	[ 1.1691102e+00]	 2.1636534e-01


.. parsed-literal::

      55	 1.2651594e+00	 1.3425821e-01	 1.3152185e+00	 1.5461230e-01	[ 1.1804062e+00]	 2.0452976e-01


.. parsed-literal::

      56	 1.2765218e+00	 1.3428550e-01	 1.3271375e+00	 1.5561477e-01	[ 1.1901185e+00]	 2.0917559e-01


.. parsed-literal::

      57	 1.2846345e+00	 1.3478705e-01	 1.3356359e+00	 1.5752552e-01	  1.1898484e+00 	 2.0615530e-01


.. parsed-literal::

      58	 1.2932843e+00	 1.3376367e-01	 1.3441045e+00	 1.5682712e-01	[ 1.1991189e+00]	 2.0677757e-01


.. parsed-literal::

      59	 1.3019647e+00	 1.3243346e-01	 1.3528233e+00	 1.5628251e-01	[ 1.2025492e+00]	 2.1342659e-01


.. parsed-literal::

      60	 1.3098536e+00	 1.3146775e-01	 1.3609230e+00	 1.5595238e-01	  1.1997081e+00 	 2.1440840e-01


.. parsed-literal::

      61	 1.3179950e+00	 1.3033018e-01	 1.3694155e+00	 1.5506835e-01	  1.1967801e+00 	 2.1406364e-01


.. parsed-literal::

      62	 1.3276503e+00	 1.2961828e-01	 1.3791369e+00	 1.5458591e-01	  1.1959695e+00 	 2.0765114e-01


.. parsed-literal::

      63	 1.3323747e+00	 1.2928843e-01	 1.3839338e+00	 1.5407489e-01	  1.1990193e+00 	 2.0613313e-01


.. parsed-literal::

      64	 1.3396976e+00	 1.2903071e-01	 1.3913578e+00	 1.5364370e-01	  1.2018179e+00 	 2.0984554e-01
      65	 1.3461702e+00	 1.2817523e-01	 1.3980732e+00	 1.5340212e-01	[ 1.2048944e+00]	 1.7654967e-01


.. parsed-literal::

      66	 1.3537193e+00	 1.2783336e-01	 1.4055259e+00	 1.5318081e-01	[ 1.2113431e+00]	 2.0651221e-01


.. parsed-literal::

      67	 1.3606108e+00	 1.2735488e-01	 1.4125829e+00	 1.5339540e-01	  1.2097820e+00 	 2.1224689e-01
      68	 1.3665363e+00	 1.2720656e-01	 1.4186917e+00	 1.5332653e-01	[ 1.2127531e+00]	 1.7740417e-01


.. parsed-literal::

      69	 1.3749700e+00	 1.2698318e-01	 1.4276247e+00	 1.5448735e-01	  1.2042010e+00 	 2.0872307e-01
      70	 1.3809386e+00	 1.2682211e-01	 1.4338066e+00	 1.5513375e-01	[ 1.2131530e+00]	 1.9994378e-01


.. parsed-literal::

      71	 1.3856849e+00	 1.2661659e-01	 1.4383033e+00	 1.5513690e-01	[ 1.2202026e+00]	 2.0070505e-01


.. parsed-literal::

      72	 1.3893730e+00	 1.2653392e-01	 1.4420152e+00	 1.5523150e-01	  1.2199486e+00 	 2.1113324e-01
      73	 1.3955617e+00	 1.2626318e-01	 1.4482169e+00	 1.5519798e-01	[ 1.2230896e+00]	 2.0138288e-01


.. parsed-literal::

      74	 1.4012628e+00	 1.2666746e-01	 1.4540030e+00	 1.5491562e-01	  1.2214623e+00 	 2.0452785e-01


.. parsed-literal::

      75	 1.4066958e+00	 1.2634684e-01	 1.4593532e+00	 1.5506259e-01	  1.2227980e+00 	 2.1587324e-01


.. parsed-literal::

      76	 1.4099824e+00	 1.2613210e-01	 1.4626446e+00	 1.5502106e-01	[ 1.2245856e+00]	 2.1508551e-01
      77	 1.4146176e+00	 1.2599196e-01	 1.4675361e+00	 1.5524202e-01	  1.2163935e+00 	 1.9725108e-01


.. parsed-literal::

      78	 1.4180798e+00	 1.2603133e-01	 1.4714051e+00	 1.5574738e-01	  1.2099886e+00 	 2.1048975e-01


.. parsed-literal::

      79	 1.4232729e+00	 1.2586949e-01	 1.4764479e+00	 1.5543435e-01	  1.2081701e+00 	 2.1142817e-01
      80	 1.4262730e+00	 1.2570433e-01	 1.4794218e+00	 1.5529102e-01	  1.2077100e+00 	 2.0165777e-01


.. parsed-literal::

      81	 1.4294656e+00	 1.2544719e-01	 1.4826306e+00	 1.5491672e-01	  1.2053514e+00 	 2.0745039e-01


.. parsed-literal::

      82	 1.4338807e+00	 1.2494871e-01	 1.4872270e+00	 1.5449504e-01	  1.2037501e+00 	 2.1899486e-01
      83	 1.4382920e+00	 1.2472179e-01	 1.4916784e+00	 1.5433554e-01	  1.1927251e+00 	 1.7010427e-01


.. parsed-literal::

      84	 1.4409754e+00	 1.2476469e-01	 1.4943398e+00	 1.5437312e-01	  1.1928916e+00 	 2.1149063e-01


.. parsed-literal::

      85	 1.4439577e+00	 1.2480855e-01	 1.4974183e+00	 1.5468949e-01	  1.1880161e+00 	 2.0371103e-01


.. parsed-literal::

      86	 1.4468829e+00	 1.2502752e-01	 1.5005413e+00	 1.5488471e-01	  1.1812502e+00 	 2.2033095e-01


.. parsed-literal::

      87	 1.4500488e+00	 1.2486607e-01	 1.5035913e+00	 1.5499770e-01	  1.1825931e+00 	 2.1119809e-01


.. parsed-literal::

      88	 1.4524652e+00	 1.2464809e-01	 1.5059358e+00	 1.5497006e-01	  1.1818002e+00 	 2.0559645e-01


.. parsed-literal::

      89	 1.4543226e+00	 1.2453641e-01	 1.5077249e+00	 1.5489872e-01	  1.1806974e+00 	 2.1496630e-01
      90	 1.4584207e+00	 1.2419740e-01	 1.5118084e+00	 1.5478415e-01	  1.1712246e+00 	 1.9141078e-01


.. parsed-literal::

      91	 1.4608371e+00	 1.2418271e-01	 1.5142871e+00	 1.5468913e-01	  1.1624176e+00 	 1.9914603e-01
      92	 1.4629843e+00	 1.2414286e-01	 1.5164028e+00	 1.5465153e-01	  1.1654984e+00 	 1.9914317e-01


.. parsed-literal::

      93	 1.4651642e+00	 1.2408516e-01	 1.5186567e+00	 1.5472150e-01	  1.1658807e+00 	 2.0594096e-01


.. parsed-literal::

      94	 1.4679033e+00	 1.2395134e-01	 1.5214659e+00	 1.5490319e-01	  1.1678779e+00 	 2.0876932e-01


.. parsed-literal::

      95	 1.4696596e+00	 1.2380569e-01	 1.5234256e+00	 1.5516131e-01	  1.1699050e+00 	 2.1033192e-01
      96	 1.4730737e+00	 1.2374180e-01	 1.5266535e+00	 1.5543565e-01	  1.1734618e+00 	 1.7745185e-01


.. parsed-literal::

      97	 1.4745713e+00	 1.2370638e-01	 1.5280572e+00	 1.5547796e-01	  1.1755114e+00 	 2.1240687e-01
      98	 1.4769079e+00	 1.2362092e-01	 1.5303483e+00	 1.5558607e-01	  1.1736772e+00 	 1.7788434e-01


.. parsed-literal::

      99	 1.4806185e+00	 1.2339062e-01	 1.5341089e+00	 1.5547622e-01	  1.1650308e+00 	 1.8488622e-01


.. parsed-literal::

     100	 1.4833223e+00	 1.2326186e-01	 1.5369826e+00	 1.5568022e-01	  1.1430867e+00 	 2.0785570e-01


.. parsed-literal::

     101	 1.4858676e+00	 1.2305490e-01	 1.5395390e+00	 1.5527773e-01	  1.1453218e+00 	 2.1023536e-01


.. parsed-literal::

     102	 1.4874812e+00	 1.2295707e-01	 1.5412143e+00	 1.5488816e-01	  1.1455689e+00 	 2.0778060e-01


.. parsed-literal::

     103	 1.4892554e+00	 1.2276611e-01	 1.5431349e+00	 1.5460532e-01	  1.1414284e+00 	 2.1510935e-01
     104	 1.4917589e+00	 1.2266886e-01	 1.5457479e+00	 1.5430824e-01	  1.1369885e+00 	 1.9735694e-01


.. parsed-literal::

     105	 1.4943685e+00	 1.2261363e-01	 1.5484930e+00	 1.5421090e-01	  1.1329240e+00 	 2.1626425e-01
     106	 1.4959283e+00	 1.2245353e-01	 1.5500634e+00	 1.5427108e-01	  1.1295981e+00 	 1.9393516e-01


.. parsed-literal::

     107	 1.4973840e+00	 1.2242012e-01	 1.5514744e+00	 1.5426841e-01	  1.1331981e+00 	 1.9806480e-01
     108	 1.4998944e+00	 1.2231917e-01	 1.5540300e+00	 1.5434559e-01	  1.1333865e+00 	 1.9026923e-01


.. parsed-literal::

     109	 1.5015494e+00	 1.2230533e-01	 1.5558073e+00	 1.5422641e-01	  1.1338348e+00 	 2.1458817e-01


.. parsed-literal::

     110	 1.5034152e+00	 1.2229529e-01	 1.5577529e+00	 1.5425915e-01	  1.1309070e+00 	 2.1192336e-01


.. parsed-literal::

     111	 1.5057240e+00	 1.2230327e-01	 1.5601679e+00	 1.5425898e-01	  1.1275394e+00 	 2.2044110e-01


.. parsed-literal::

     112	 1.5077832e+00	 1.2236627e-01	 1.5622410e+00	 1.5428934e-01	  1.1265473e+00 	 2.1360636e-01


.. parsed-literal::

     113	 1.5101587e+00	 1.2235318e-01	 1.5646643e+00	 1.5405411e-01	  1.1268665e+00 	 2.0964432e-01


.. parsed-literal::

     114	 1.5120468e+00	 1.2233637e-01	 1.5664621e+00	 1.5410529e-01	  1.1341628e+00 	 2.0186877e-01


.. parsed-literal::

     115	 1.5134759e+00	 1.2226906e-01	 1.5677997e+00	 1.5413789e-01	  1.1370518e+00 	 2.1046090e-01


.. parsed-literal::

     116	 1.5147192e+00	 1.2223009e-01	 1.5690739e+00	 1.5420607e-01	  1.1392004e+00 	 2.2039795e-01


.. parsed-literal::

     117	 1.5163813e+00	 1.2216141e-01	 1.5707352e+00	 1.5423009e-01	  1.1375074e+00 	 2.1384668e-01


.. parsed-literal::

     118	 1.5178269e+00	 1.2216615e-01	 1.5722638e+00	 1.5425743e-01	  1.1325158e+00 	 2.0474863e-01
     119	 1.5193887e+00	 1.2216976e-01	 1.5739104e+00	 1.5423533e-01	  1.1274403e+00 	 1.8486714e-01


.. parsed-literal::

     120	 1.5210671e+00	 1.2206033e-01	 1.5757381e+00	 1.5411650e-01	  1.1223101e+00 	 2.0945430e-01
     121	 1.5233095e+00	 1.2198934e-01	 1.5779668e+00	 1.5406581e-01	  1.1203205e+00 	 1.8036151e-01


.. parsed-literal::

     122	 1.5248331e+00	 1.2191359e-01	 1.5794603e+00	 1.5409135e-01	  1.1222767e+00 	 1.8259883e-01
     123	 1.5267563e+00	 1.2185985e-01	 1.5814117e+00	 1.5419954e-01	  1.1234100e+00 	 1.7755723e-01


.. parsed-literal::

     124	 1.5272946e+00	 1.2199217e-01	 1.5821745e+00	 1.5462799e-01	  1.1173240e+00 	 2.1888256e-01
     125	 1.5297889e+00	 1.2194510e-01	 1.5845607e+00	 1.5449323e-01	  1.1206790e+00 	 1.8546677e-01


.. parsed-literal::

     126	 1.5307771e+00	 1.2196183e-01	 1.5855668e+00	 1.5445405e-01	  1.1199536e+00 	 1.7263198e-01
     127	 1.5322628e+00	 1.2197534e-01	 1.5871053e+00	 1.5436341e-01	  1.1165743e+00 	 1.8641472e-01


.. parsed-literal::

     128	 1.5343775e+00	 1.2188720e-01	 1.5892889e+00	 1.5415862e-01	  1.1096484e+00 	 2.1283793e-01


.. parsed-literal::

     129	 1.5354992e+00	 1.2180610e-01	 1.5905393e+00	 1.5397645e-01	  1.0979604e+00 	 3.0499649e-01


.. parsed-literal::

     130	 1.5376994e+00	 1.2164707e-01	 1.5927633e+00	 1.5376245e-01	  1.0870146e+00 	 2.1031761e-01


.. parsed-literal::

     131	 1.5391686e+00	 1.2152770e-01	 1.5942501e+00	 1.5366221e-01	  1.0806511e+00 	 2.1044040e-01
     132	 1.5404744e+00	 1.2136529e-01	 1.5955997e+00	 1.5359253e-01	  1.0710328e+00 	 1.8913198e-01


.. parsed-literal::

     133	 1.5415877e+00	 1.2137358e-01	 1.5967246e+00	 1.5365126e-01	  1.0683132e+00 	 2.1645331e-01


.. parsed-literal::

     134	 1.5424964e+00	 1.2139364e-01	 1.5976281e+00	 1.5366885e-01	  1.0688877e+00 	 2.1346807e-01


.. parsed-literal::

     135	 1.5440799e+00	 1.2146064e-01	 1.5992394e+00	 1.5378466e-01	  1.0648438e+00 	 2.0979190e-01


.. parsed-literal::

     136	 1.5451386e+00	 1.2142064e-01	 1.6003718e+00	 1.5373912e-01	  1.0570075e+00 	 2.0885873e-01


.. parsed-literal::

     137	 1.5464435e+00	 1.2142441e-01	 1.6016585e+00	 1.5387625e-01	  1.0548108e+00 	 2.0411706e-01


.. parsed-literal::

     138	 1.5477345e+00	 1.2135387e-01	 1.6029519e+00	 1.5398809e-01	  1.0482033e+00 	 2.1654081e-01


.. parsed-literal::

     139	 1.5488502e+00	 1.2124118e-01	 1.6040727e+00	 1.5401517e-01	  1.0444723e+00 	 2.0635724e-01


.. parsed-literal::

     140	 1.5510267e+00	 1.2111100e-01	 1.6063180e+00	 1.5410996e-01	  1.0375954e+00 	 2.0823836e-01


.. parsed-literal::

     141	 1.5521174e+00	 1.2087239e-01	 1.6074539e+00	 1.5391819e-01	  1.0403165e+00 	 3.2563305e-01


.. parsed-literal::

     142	 1.5531974e+00	 1.2085951e-01	 1.6085629e+00	 1.5393534e-01	  1.0404748e+00 	 2.1562338e-01
     143	 1.5542062e+00	 1.2084480e-01	 1.6096075e+00	 1.5389521e-01	  1.0425300e+00 	 1.9171596e-01


.. parsed-literal::

     144	 1.5552323e+00	 1.2084024e-01	 1.6106789e+00	 1.5388129e-01	  1.0459128e+00 	 2.0604110e-01
     145	 1.5564360e+00	 1.2083815e-01	 1.6118943e+00	 1.5388776e-01	  1.0442104e+00 	 1.8834257e-01


.. parsed-literal::

     146	 1.5575124e+00	 1.2079575e-01	 1.6129505e+00	 1.5391822e-01	  1.0433788e+00 	 2.0766234e-01


.. parsed-literal::

     147	 1.5585833e+00	 1.2078458e-01	 1.6139943e+00	 1.5396038e-01	  1.0440199e+00 	 2.1097612e-01


.. parsed-literal::

     148	 1.5599344e+00	 1.2077405e-01	 1.6153206e+00	 1.5390775e-01	  1.0431181e+00 	 2.0878458e-01


.. parsed-literal::

     149	 1.5616507e+00	 1.2092269e-01	 1.6170485e+00	 1.5394908e-01	  1.0509092e+00 	 2.0611143e-01


.. parsed-literal::

     150	 1.5624824e+00	 1.2097029e-01	 1.6178898e+00	 1.5359836e-01	  1.0472020e+00 	 2.1898293e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.11 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe884d2b9d0>



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
    CPU times: user 1.7 s, sys: 52 ms, total: 1.75 s
    Wall time: 537 ms


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

