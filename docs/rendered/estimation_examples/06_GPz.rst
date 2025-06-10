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
       1	-3.4906600e-01	 3.2198365e-01	-3.3942557e-01	 3.1441995e-01	[-3.2581927e-01]	 4.6692610e-01


.. parsed-literal::

       2	-2.7683035e-01	 3.1114324e-01	-2.5278846e-01	 3.0468297e-01	[-2.3379369e-01]	 2.3039532e-01


.. parsed-literal::

       3	-2.3372106e-01	 2.9124077e-01	-1.9251264e-01	 2.8541962e-01	[-1.7081978e-01]	 2.9716182e-01
       4	-1.9423090e-01	 2.6703797e-01	-1.5268482e-01	 2.6282222e-01	[-1.3169378e-01]	 2.0456219e-01


.. parsed-literal::

       5	-1.0991527e-01	 2.5804582e-01	-7.6571087e-02	 2.5429655e-01	[-6.0980095e-02]	 2.0211601e-01


.. parsed-literal::

       6	-7.2974624e-02	 2.5222378e-01	-4.2768522e-02	 2.4986916e-01	[-3.4681680e-02]	 2.0573807e-01


.. parsed-literal::

       7	-5.7043733e-02	 2.4982919e-01	-3.2292000e-02	 2.4712519e-01	[-2.2074121e-02]	 2.0915818e-01


.. parsed-literal::

       8	-4.1990753e-02	 2.4728010e-01	-2.1653829e-02	 2.4430912e-01	[-1.0112969e-02]	 2.0068359e-01


.. parsed-literal::

       9	-2.9699209e-02	 2.4504458e-01	-1.2237292e-02	 2.4177348e-01	[ 7.8225576e-04]	 2.0196986e-01


.. parsed-literal::

      10	-2.2106556e-02	 2.4386050e-01	-6.8662696e-03	 2.3998670e-01	[ 7.6216993e-03]	 2.0185399e-01


.. parsed-literal::

      11	-1.4659494e-02	 2.4237122e-01	-3.0702120e-04	 2.3818437e-01	[ 1.6724146e-02]	 2.0877886e-01
      12	-1.2212562e-02	 2.4194079e-01	 1.9035104e-03	 2.3761421e-01	[ 1.9039157e-02]	 1.8341851e-01


.. parsed-literal::

      13	-8.1524980e-03	 2.4122504e-01	 5.7439999e-03	 2.3660954e-01	[ 2.4346259e-02]	 2.0499372e-01


.. parsed-literal::

      14	 2.6121586e-02	 2.3213247e-01	 4.1677167e-02	 2.2459419e-01	[ 6.3724594e-02]	 3.2270956e-01


.. parsed-literal::

      15	 4.4769468e-02	 2.2640617e-01	 6.3095802e-02	 2.2110608e-01	[ 7.4122554e-02]	 2.1279597e-01


.. parsed-literal::

      16	 1.2628069e-01	 2.2194827e-01	 1.4687074e-01	 2.1950138e-01	[ 1.5156251e-01]	 2.1828508e-01


.. parsed-literal::

      17	 2.0342239e-01	 2.1866063e-01	 2.3310926e-01	 2.1666186e-01	[ 2.3588173e-01]	 2.1471596e-01


.. parsed-literal::

      18	 2.6366353e-01	 2.1852245e-01	 2.9336238e-01	 2.1544496e-01	[ 2.9955582e-01]	 2.0546341e-01


.. parsed-literal::

      19	 2.9536445e-01	 2.1627761e-01	 3.2580022e-01	 2.1294191e-01	[ 3.3162348e-01]	 2.1782827e-01


.. parsed-literal::

      20	 3.3988745e-01	 2.1552980e-01	 3.6969849e-01	 2.1349287e-01	[ 3.6997649e-01]	 2.0711923e-01


.. parsed-literal::

      21	 4.2205003e-01	 2.1321205e-01	 4.5375668e-01	 2.1124106e-01	[ 4.5079876e-01]	 2.1661401e-01


.. parsed-literal::

      22	 4.8893342e-01	 2.1448565e-01	 5.2186619e-01	 2.1316565e-01	[ 5.0950132e-01]	 2.0709348e-01


.. parsed-literal::

      23	 5.5742713e-01	 2.1047885e-01	 5.9335724e-01	 2.1014572e-01	[ 5.7389856e-01]	 2.0855832e-01
      24	 5.8994291e-01	 2.0762357e-01	 6.2885325e-01	 2.0715490e-01	[ 5.9583322e-01]	 1.9244528e-01


.. parsed-literal::

      25	 6.2928778e-01	 2.0443936e-01	 6.6820550e-01	 2.0326455e-01	[ 6.3686917e-01]	 2.1678615e-01


.. parsed-literal::

      26	 6.9197353e-01	 2.0156654e-01	 7.2893572e-01	 1.9925906e-01	[ 7.0565252e-01]	 2.0841193e-01


.. parsed-literal::

      27	 7.2159394e-01	 1.9930823e-01	 7.5621449e-01	 1.9568360e-01	[ 7.4632001e-01]	 2.0603824e-01
      28	 7.5592973e-01	 1.9840534e-01	 7.9282683e-01	 1.9521078e-01	[ 7.7104113e-01]	 1.7780733e-01


.. parsed-literal::

      29	 7.7691755e-01	 1.9748937e-01	 8.1537345e-01	 1.9380990e-01	[ 7.8396916e-01]	 2.1102142e-01
      30	 7.9924537e-01	 1.9687486e-01	 8.3834594e-01	 1.9312876e-01	[ 8.0645591e-01]	 1.9175553e-01


.. parsed-literal::

      31	 8.2767469e-01	 1.9705621e-01	 8.6728948e-01	 1.9294645e-01	[ 8.4553573e-01]	 2.1586394e-01


.. parsed-literal::

      32	 8.6926387e-01	 1.9629413e-01	 9.1133728e-01	 1.9198877e-01	[ 8.8362543e-01]	 2.0738006e-01
      33	 9.0213009e-01	 1.9737365e-01	 9.4580622e-01	 1.9091413e-01	[ 9.2861003e-01]	 2.0181084e-01


.. parsed-literal::

      34	 9.2916317e-01	 1.9228435e-01	 9.7498380e-01	 1.8542602e-01	[ 9.3818834e-01]	 2.0022798e-01
      35	 9.5344561e-01	 1.8902612e-01	 9.9698054e-01	 1.8427658e-01	[ 9.7315485e-01]	 1.9832325e-01


.. parsed-literal::

      36	 9.6343516e-01	 1.8706485e-01	 1.0066674e+00	 1.8248950e-01	[ 9.8321627e-01]	 1.9866252e-01


.. parsed-literal::

      37	 9.7350855e-01	 1.8490247e-01	 1.0168159e+00	 1.8074559e-01	[ 9.8844633e-01]	 2.0823097e-01


.. parsed-literal::

      38	 9.8388634e-01	 1.8281702e-01	 1.0278024e+00	 1.7839566e-01	  9.8535226e-01 	 2.1182275e-01
      39	 9.9289808e-01	 1.8130105e-01	 1.0370128e+00	 1.7699001e-01	  9.8639823e-01 	 1.9197488e-01


.. parsed-literal::

      40	 9.9870699e-01	 1.8013546e-01	 1.0429542e+00	 1.7605059e-01	[ 9.8998073e-01]	 2.1289229e-01


.. parsed-literal::

      41	 1.0130187e+00	 1.7735442e-01	 1.0577798e+00	 1.7383683e-01	[ 9.9919633e-01]	 2.0460725e-01
      42	 1.0183288e+00	 1.7722095e-01	 1.0636162e+00	 1.7338954e-01	[ 1.0025567e+00]	 1.9682717e-01


.. parsed-literal::

      43	 1.0301336e+00	 1.7527389e-01	 1.0752828e+00	 1.7185505e-01	[ 1.0140225e+00]	 2.0266151e-01


.. parsed-literal::

      44	 1.0371127e+00	 1.7481317e-01	 1.0822340e+00	 1.7186056e-01	[ 1.0216297e+00]	 2.1665382e-01
      45	 1.0444681e+00	 1.7439118e-01	 1.0897263e+00	 1.7187329e-01	[ 1.0256023e+00]	 1.9489503e-01


.. parsed-literal::

      46	 1.0582421e+00	 1.7383728e-01	 1.1039069e+00	 1.7281198e-01	[ 1.0359123e+00]	 1.9211864e-01


.. parsed-literal::

      47	 1.0637442e+00	 1.7452725e-01	 1.1100420e+00	 1.7425215e-01	  1.0303999e+00 	 2.1689701e-01


.. parsed-literal::

      48	 1.0721654e+00	 1.7334730e-01	 1.1182073e+00	 1.7306828e-01	[ 1.0410071e+00]	 2.1065593e-01


.. parsed-literal::

      49	 1.0789159e+00	 1.7239224e-01	 1.1252067e+00	 1.7239692e-01	[ 1.0431813e+00]	 2.1280003e-01


.. parsed-literal::

      50	 1.0845451e+00	 1.7197534e-01	 1.1310117e+00	 1.7226107e-01	[ 1.0455379e+00]	 2.1596742e-01
      51	 1.0941132e+00	 1.7074657e-01	 1.1410618e+00	 1.7231308e-01	[ 1.0513043e+00]	 1.9194126e-01


.. parsed-literal::

      52	 1.1039371e+00	 1.6959397e-01	 1.1510188e+00	 1.7240139e-01	[ 1.0732336e+00]	 2.1566033e-01


.. parsed-literal::

      53	 1.1102025e+00	 1.6871541e-01	 1.1571962e+00	 1.7186535e-01	[ 1.0863296e+00]	 2.0337176e-01


.. parsed-literal::

      54	 1.1207543e+00	 1.6699825e-01	 1.1679448e+00	 1.7101928e-01	[ 1.1050918e+00]	 2.1281409e-01


.. parsed-literal::

      55	 1.1294487e+00	 1.6608567e-01	 1.1768853e+00	 1.7049713e-01	[ 1.1158476e+00]	 2.2130752e-01


.. parsed-literal::

      56	 1.1397639e+00	 1.6443749e-01	 1.1875273e+00	 1.6943962e-01	[ 1.1280429e+00]	 2.2651076e-01


.. parsed-literal::

      57	 1.1484704e+00	 1.6303710e-01	 1.1964909e+00	 1.6870396e-01	[ 1.1378219e+00]	 2.2144341e-01


.. parsed-literal::

      58	 1.1587066e+00	 1.6183500e-01	 1.2068434e+00	 1.6773937e-01	[ 1.1492950e+00]	 2.1604276e-01
      59	 1.1723424e+00	 1.5972627e-01	 1.2208932e+00	 1.6603582e-01	[ 1.1570838e+00]	 1.9979811e-01


.. parsed-literal::

      60	 1.1821494e+00	 1.5956475e-01	 1.2313221e+00	 1.6582189e-01	[ 1.1642582e+00]	 1.8452716e-01


.. parsed-literal::

      61	 1.1922085e+00	 1.5844342e-01	 1.2409875e+00	 1.6458416e-01	[ 1.1730813e+00]	 2.1599364e-01


.. parsed-literal::

      62	 1.1997975e+00	 1.5845729e-01	 1.2487660e+00	 1.6439294e-01	[ 1.1775324e+00]	 2.1908998e-01


.. parsed-literal::

      63	 1.2061763e+00	 1.5805254e-01	 1.2554902e+00	 1.6365723e-01	[ 1.1806117e+00]	 2.0456457e-01


.. parsed-literal::

      64	 1.2163911e+00	 1.5781754e-01	 1.2661839e+00	 1.6301525e-01	[ 1.1921266e+00]	 2.1502995e-01
      65	 1.2206681e+00	 1.5730342e-01	 1.2710620e+00	 1.6185553e-01	[ 1.1957580e+00]	 1.8072009e-01


.. parsed-literal::

      66	 1.2304764e+00	 1.5637585e-01	 1.2805219e+00	 1.6117956e-01	[ 1.2064939e+00]	 2.2524357e-01


.. parsed-literal::

      67	 1.2352793e+00	 1.5590835e-01	 1.2852916e+00	 1.6106392e-01	[ 1.2116621e+00]	 2.1085024e-01
      68	 1.2402927e+00	 1.5498186e-01	 1.2903751e+00	 1.6025510e-01	[ 1.2150303e+00]	 1.7874694e-01


.. parsed-literal::

      69	 1.2493888e+00	 1.5314134e-01	 1.3000156e+00	 1.5814544e-01	[ 1.2190382e+00]	 1.9897032e-01
      70	 1.2566841e+00	 1.5210351e-01	 1.3076965e+00	 1.5698212e-01	[ 1.2197937e+00]	 1.9991994e-01


.. parsed-literal::

      71	 1.2628213e+00	 1.5148453e-01	 1.3137622e+00	 1.5628584e-01	[ 1.2236650e+00]	 1.9319868e-01
      72	 1.2722061e+00	 1.5107060e-01	 1.3235140e+00	 1.5507094e-01	[ 1.2270568e+00]	 2.0157051e-01


.. parsed-literal::

      73	 1.2788658e+00	 1.5053460e-01	 1.3302893e+00	 1.5398463e-01	[ 1.2355651e+00]	 1.8785119e-01
      74	 1.2882670e+00	 1.5027573e-01	 1.3398687e+00	 1.5314813e-01	[ 1.2421461e+00]	 1.9859791e-01


.. parsed-literal::

      75	 1.2954436e+00	 1.5001685e-01	 1.3473664e+00	 1.5239980e-01	[ 1.2482920e+00]	 2.1313500e-01


.. parsed-literal::

      76	 1.3013189e+00	 1.4945532e-01	 1.3532102e+00	 1.5194690e-01	[ 1.2483140e+00]	 2.2449112e-01


.. parsed-literal::

      77	 1.3054887e+00	 1.4919245e-01	 1.3573128e+00	 1.5177150e-01	[ 1.2491814e+00]	 2.0947146e-01


.. parsed-literal::

      78	 1.3120709e+00	 1.4877772e-01	 1.3639014e+00	 1.5165142e-01	  1.2439970e+00 	 2.1174169e-01
      79	 1.3185558e+00	 1.4868806e-01	 1.3707030e+00	 1.5152358e-01	  1.2402265e+00 	 2.0546818e-01


.. parsed-literal::

      80	 1.3244787e+00	 1.4839692e-01	 1.3765504e+00	 1.5106558e-01	  1.2401469e+00 	 2.0238113e-01


.. parsed-literal::

      81	 1.3300111e+00	 1.4799588e-01	 1.3821190e+00	 1.5015091e-01	  1.2414229e+00 	 2.2379804e-01


.. parsed-literal::

      82	 1.3335743e+00	 1.4739170e-01	 1.3857879e+00	 1.4963150e-01	  1.2390032e+00 	 2.1990919e-01


.. parsed-literal::

      83	 1.3389709e+00	 1.4630791e-01	 1.3913433e+00	 1.4855595e-01	  1.2386810e+00 	 2.0640302e-01
      84	 1.3478234e+00	 1.4397923e-01	 1.4003997e+00	 1.4646455e-01	[ 1.2503582e+00]	 1.8873644e-01


.. parsed-literal::

      85	 1.3504730e+00	 1.4106916e-01	 1.4034609e+00	 1.4390513e-01	  1.2390980e+00 	 2.1724463e-01


.. parsed-literal::

      86	 1.3559623e+00	 1.4218816e-01	 1.4085417e+00	 1.4507247e-01	[ 1.2555690e+00]	 2.2249246e-01


.. parsed-literal::

      87	 1.3587426e+00	 1.4225022e-01	 1.4112959e+00	 1.4517606e-01	[ 1.2602195e+00]	 2.0758104e-01
      88	 1.3645849e+00	 1.4196966e-01	 1.4172329e+00	 1.4512249e-01	[ 1.2683003e+00]	 1.9343805e-01


.. parsed-literal::

      89	 1.3687000e+00	 1.4138564e-01	 1.4216521e+00	 1.4429128e-01	  1.2673048e+00 	 2.0879602e-01


.. parsed-literal::

      90	 1.3742941e+00	 1.4109512e-01	 1.4272516e+00	 1.4424139e-01	[ 1.2697821e+00]	 2.0782495e-01


.. parsed-literal::

      91	 1.3785347e+00	 1.4079640e-01	 1.4315350e+00	 1.4403351e-01	  1.2692287e+00 	 2.0882583e-01
      92	 1.3822640e+00	 1.4038401e-01	 1.4353400e+00	 1.4360706e-01	  1.2687697e+00 	 2.0631957e-01


.. parsed-literal::

      93	 1.3835343e+00	 1.3994692e-01	 1.4369741e+00	 1.4323375e-01	  1.2670823e+00 	 2.0117378e-01
      94	 1.3902936e+00	 1.3899443e-01	 1.4436340e+00	 1.4228701e-01	[ 1.2699073e+00]	 1.9630241e-01


.. parsed-literal::

      95	 1.3922874e+00	 1.3883775e-01	 1.4455767e+00	 1.4215375e-01	[ 1.2738677e+00]	 2.0792937e-01


.. parsed-literal::

      96	 1.3953913e+00	 1.3831485e-01	 1.4487615e+00	 1.4179606e-01	[ 1.2774545e+00]	 2.1096659e-01


.. parsed-literal::

      97	 1.3992086e+00	 1.3768094e-01	 1.4526731e+00	 1.4153519e-01	[ 1.2792301e+00]	 2.0994687e-01
      98	 1.4045040e+00	 1.3655489e-01	 1.4582847e+00	 1.4124361e-01	[ 1.2812732e+00]	 2.1036410e-01


.. parsed-literal::

      99	 1.4097526e+00	 1.3483278e-01	 1.4635740e+00	 1.4041937e-01	  1.2761039e+00 	 2.1297169e-01


.. parsed-literal::

     100	 1.4131380e+00	 1.3489834e-01	 1.4667981e+00	 1.4050200e-01	  1.2803683e+00 	 2.1004963e-01


.. parsed-literal::

     101	 1.4185305e+00	 1.3402110e-01	 1.4721795e+00	 1.3992572e-01	  1.2771389e+00 	 2.0804191e-01
     102	 1.4201955e+00	 1.3361156e-01	 1.4739062e+00	 1.3983812e-01	[ 1.2847626e+00]	 2.0158958e-01


.. parsed-literal::

     103	 1.4237886e+00	 1.3321867e-01	 1.4774330e+00	 1.3944799e-01	  1.2842679e+00 	 2.1628952e-01


.. parsed-literal::

     104	 1.4264963e+00	 1.3265359e-01	 1.4801417e+00	 1.3892778e-01	  1.2838547e+00 	 2.1373725e-01
     105	 1.4293422e+00	 1.3210103e-01	 1.4830257e+00	 1.3844743e-01	  1.2833771e+00 	 1.9197607e-01


.. parsed-literal::

     106	 1.4322653e+00	 1.3161127e-01	 1.4861054e+00	 1.3797487e-01	[ 1.2865021e+00]	 2.0980740e-01


.. parsed-literal::

     107	 1.4365256e+00	 1.3121306e-01	 1.4903286e+00	 1.3758728e-01	  1.2857675e+00 	 2.0691204e-01
     108	 1.4387182e+00	 1.3117557e-01	 1.4925209e+00	 1.3758401e-01	[ 1.2865625e+00]	 1.9938993e-01


.. parsed-literal::

     109	 1.4411051e+00	 1.3108495e-01	 1.4949605e+00	 1.3742220e-01	  1.2863544e+00 	 1.7248678e-01


.. parsed-literal::

     110	 1.4443521e+00	 1.3083467e-01	 1.4983024e+00	 1.3722241e-01	[ 1.2879062e+00]	 2.1185160e-01
     111	 1.4472577e+00	 1.3028005e-01	 1.5013270e+00	 1.3640941e-01	  1.2831053e+00 	 1.9593835e-01


.. parsed-literal::

     112	 1.4495705e+00	 1.3025070e-01	 1.5035831e+00	 1.3639924e-01	  1.2871640e+00 	 1.7228127e-01
     113	 1.4515889e+00	 1.2987557e-01	 1.5055870e+00	 1.3601207e-01	[ 1.2898031e+00]	 1.8802381e-01


.. parsed-literal::

     114	 1.4537114e+00	 1.2964242e-01	 1.5077221e+00	 1.3569131e-01	[ 1.2939196e+00]	 2.1526122e-01
     115	 1.4567889e+00	 1.2876448e-01	 1.5109197e+00	 1.3474462e-01	[ 1.2948184e+00]	 1.8950677e-01


.. parsed-literal::

     116	 1.4595231e+00	 1.2846716e-01	 1.5136178e+00	 1.3429644e-01	[ 1.3001628e+00]	 2.1901178e-01
     117	 1.4619308e+00	 1.2819955e-01	 1.5159886e+00	 1.3404732e-01	[ 1.3018243e+00]	 1.7788339e-01


.. parsed-literal::

     118	 1.4640405e+00	 1.2789038e-01	 1.5180842e+00	 1.3370703e-01	  1.2984597e+00 	 2.1024966e-01
     119	 1.4663714e+00	 1.2756152e-01	 1.5204049e+00	 1.3352558e-01	  1.2986345e+00 	 1.9532156e-01


.. parsed-literal::

     120	 1.4682865e+00	 1.2746542e-01	 1.5223038e+00	 1.3358492e-01	  1.2970966e+00 	 1.9701409e-01


.. parsed-literal::

     121	 1.4706555e+00	 1.2731681e-01	 1.5247176e+00	 1.3352717e-01	  1.2975234e+00 	 2.1004772e-01


.. parsed-literal::

     122	 1.4731743e+00	 1.2691135e-01	 1.5273419e+00	 1.3321084e-01	  1.2968627e+00 	 2.1015000e-01


.. parsed-literal::

     123	 1.4761102e+00	 1.2671498e-01	 1.5303708e+00	 1.3302133e-01	  1.3008581e+00 	 2.0949817e-01


.. parsed-literal::

     124	 1.4786192e+00	 1.2630508e-01	 1.5329179e+00	 1.3258932e-01	[ 1.3033641e+00]	 2.0195222e-01
     125	 1.4810459e+00	 1.2605965e-01	 1.5354060e+00	 1.3235295e-01	[ 1.3079615e+00]	 1.9816875e-01


.. parsed-literal::

     126	 1.4828258e+00	 1.2554828e-01	 1.5372584e+00	 1.3199015e-01	  1.3051261e+00 	 2.0614576e-01


.. parsed-literal::

     127	 1.4846376e+00	 1.2543731e-01	 1.5390641e+00	 1.3196602e-01	[ 1.3087467e+00]	 2.1747613e-01


.. parsed-literal::

     128	 1.4864798e+00	 1.2522078e-01	 1.5409549e+00	 1.3194925e-01	[ 1.3105764e+00]	 2.1121407e-01
     129	 1.4878411e+00	 1.2508561e-01	 1.5423809e+00	 1.3191892e-01	[ 1.3128030e+00]	 1.9700003e-01


.. parsed-literal::

     130	 1.4894982e+00	 1.2480885e-01	 1.5440862e+00	 1.3179888e-01	  1.3089260e+00 	 2.0022941e-01


.. parsed-literal::

     131	 1.4909582e+00	 1.2466321e-01	 1.5455762e+00	 1.3177108e-01	  1.3078031e+00 	 2.0415306e-01


.. parsed-literal::

     132	 1.4925995e+00	 1.2450466e-01	 1.5472752e+00	 1.3165934e-01	  1.3004669e+00 	 2.1950889e-01


.. parsed-literal::

     133	 1.4944544e+00	 1.2445339e-01	 1.5491421e+00	 1.3169709e-01	  1.2985376e+00 	 2.1055222e-01


.. parsed-literal::

     134	 1.4967114e+00	 1.2443205e-01	 1.5514343e+00	 1.3176428e-01	  1.2939733e+00 	 2.0457458e-01


.. parsed-literal::

     135	 1.4979580e+00	 1.2458171e-01	 1.5527413e+00	 1.3181156e-01	  1.2933302e+00 	 2.1151161e-01
     136	 1.4993610e+00	 1.2449243e-01	 1.5541072e+00	 1.3172805e-01	  1.2941898e+00 	 1.9034314e-01


.. parsed-literal::

     137	 1.5010931e+00	 1.2435271e-01	 1.5558492e+00	 1.3151206e-01	  1.2936694e+00 	 1.9967628e-01


.. parsed-literal::

     138	 1.5027662e+00	 1.2417225e-01	 1.5575737e+00	 1.3120408e-01	  1.2891132e+00 	 2.1225905e-01


.. parsed-literal::

     139	 1.5047532e+00	 1.2404847e-01	 1.5597342e+00	 1.3081595e-01	  1.2860717e+00 	 2.1932411e-01


.. parsed-literal::

     140	 1.5071900e+00	 1.2383848e-01	 1.5621688e+00	 1.3050710e-01	  1.2737814e+00 	 2.1346855e-01


.. parsed-literal::

     141	 1.5086743e+00	 1.2370844e-01	 1.5636242e+00	 1.3039961e-01	  1.2708263e+00 	 2.1475649e-01


.. parsed-literal::

     142	 1.5105385e+00	 1.2358759e-01	 1.5655249e+00	 1.3040950e-01	  1.2684806e+00 	 2.0683002e-01


.. parsed-literal::

     143	 1.5114739e+00	 1.2336272e-01	 1.5665880e+00	 1.3015437e-01	  1.2597032e+00 	 2.1104050e-01
     144	 1.5131346e+00	 1.2339581e-01	 1.5681358e+00	 1.3024610e-01	  1.2678161e+00 	 1.9227862e-01


.. parsed-literal::

     145	 1.5140915e+00	 1.2337141e-01	 1.5690713e+00	 1.3026310e-01	  1.2703862e+00 	 2.1454787e-01


.. parsed-literal::

     146	 1.5151660e+00	 1.2327258e-01	 1.5701460e+00	 1.3023977e-01	  1.2720647e+00 	 2.0208454e-01


.. parsed-literal::

     147	 1.5170127e+00	 1.2315584e-01	 1.5720338e+00	 1.3021658e-01	  1.2774756e+00 	 2.1665835e-01


.. parsed-literal::

     148	 1.5177230e+00	 1.2293260e-01	 1.5728533e+00	 1.3015987e-01	  1.2629434e+00 	 2.0108271e-01
     149	 1.5192851e+00	 1.2296862e-01	 1.5743331e+00	 1.3012549e-01	  1.2695479e+00 	 1.7606950e-01


.. parsed-literal::

     150	 1.5200828e+00	 1.2297214e-01	 1.5751357e+00	 1.3007979e-01	  1.2688248e+00 	 2.0897055e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.07 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f97161bd600>



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
    CPU times: user 1.8 s, sys: 54 ms, total: 1.85 s
    Wall time: 627 ms


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

