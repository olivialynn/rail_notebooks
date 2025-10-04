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
       1	-3.4333071e-01	 3.2066988e-01	-3.3359386e-01	 3.2210690e-01	[-3.3322933e-01]	 4.6877193e-01


.. parsed-literal::

       2	-2.7378235e-01	 3.1021481e-01	-2.4937587e-01	 3.1138320e-01	[-2.4951848e-01]	 2.3885632e-01


.. parsed-literal::

       3	-2.2912540e-01	 2.8861191e-01	-1.8527797e-01	 2.9752557e-01	[-2.1220353e-01]	 2.9047847e-01


.. parsed-literal::

       4	-1.9428767e-01	 2.6588359e-01	-1.5081999e-01	 2.7730586e-01	[-2.0580455e-01]	 2.1767163e-01
       5	-1.0687941e-01	 2.5775029e-01	-7.3042187e-02	 2.6887556e-01	[-1.0946298e-01]	 1.9910574e-01


.. parsed-literal::

       6	-6.9014304e-02	 2.5112779e-01	-3.9116532e-02	 2.5682318e-01	[-5.8992650e-02]	 2.1101475e-01


.. parsed-literal::

       7	-5.2742612e-02	 2.4873504e-01	-2.8198651e-02	 2.5465669e-01	[-4.9076526e-02]	 2.0900631e-01
       8	-3.6612660e-02	 2.4589719e-01	-1.6304918e-02	 2.5176742e-01	[-3.6774696e-02]	 2.0183039e-01


.. parsed-literal::

       9	-2.2160062e-02	 2.4316584e-01	-4.5737185e-03	 2.4870043e-01	[-2.4089090e-02]	 2.1180105e-01


.. parsed-literal::

      10	-9.7429822e-03	 2.4077865e-01	 5.7161798e-03	 2.4692847e-01	[-1.7735096e-02]	 2.1690965e-01


.. parsed-literal::

      11	-5.4499712e-03	 2.4034657e-01	 8.5889397e-03	 2.4665661e-01	[-1.3962484e-02]	 2.1036148e-01


.. parsed-literal::

      12	-2.1734126e-03	 2.3964574e-01	 1.1597969e-02	 2.4608635e-01	[-1.1608716e-02]	 2.0335031e-01
      13	 9.6934250e-04	 2.3903176e-01	 1.4678506e-02	 2.4574418e-01	[-9.3264825e-03]	 1.9686699e-01


.. parsed-literal::

      14	 3.0790512e-02	 2.3331216e-01	 4.6638497e-02	 2.4719805e-01	[ 1.5091771e-02]	 2.9629016e-01


.. parsed-literal::

      15	 7.2315433e-02	 2.2504875e-01	 9.0060214e-02	 2.4359210e-01	[ 6.7879588e-02]	 3.3081031e-01
      16	 1.5982201e-01	 2.2103840e-01	 1.8348603e-01	 2.5074062e-01	[ 1.5899234e-01]	 1.9143820e-01


.. parsed-literal::

      17	 2.7856482e-01	 2.2353982e-01	 3.0744152e-01	 2.3115909e-01	[ 2.9262671e-01]	 2.1838450e-01


.. parsed-literal::

      18	 3.4412291e-01	 2.1535144e-01	 3.7435780e-01	 2.2446465e-01	[ 3.4729592e-01]	 2.1135831e-01
      19	 4.0253082e-01	 2.0914566e-01	 4.3456199e-01	 2.2043501e-01	[ 3.9940886e-01]	 1.9351697e-01


.. parsed-literal::

      20	 4.7094980e-01	 2.0210533e-01	 5.0498640e-01	 2.1398299e-01	[ 4.6520571e-01]	 2.1334314e-01


.. parsed-literal::

      21	 5.5954241e-01	 1.9606865e-01	 5.9728939e-01	 2.0784167e-01	[ 5.5078133e-01]	 2.1299911e-01
      22	 6.0318669e-01	 1.9632262e-01	 6.4193936e-01	 2.0880738e-01	[ 5.7652414e-01]	 1.9988918e-01


.. parsed-literal::

      23	 6.3504162e-01	 1.9550459e-01	 6.7346655e-01	 2.0890046e-01	[ 6.2285174e-01]	 1.9461560e-01


.. parsed-literal::

      24	 6.5609445e-01	 1.9441839e-01	 6.9414779e-01	 2.0889866e-01	[ 6.4095677e-01]	 2.1328044e-01


.. parsed-literal::

      25	 6.8085000e-01	 1.9460862e-01	 7.1795913e-01	 2.0989986e-01	[ 6.6241097e-01]	 2.1392059e-01


.. parsed-literal::

      26	 7.1119302e-01	 1.9430315e-01	 7.4827581e-01	 2.0858553e-01	[ 6.9804429e-01]	 2.0938540e-01


.. parsed-literal::

      27	 7.4074377e-01	 1.9424608e-01	 7.7932830e-01	 2.0919884e-01	[ 7.2585181e-01]	 2.0439100e-01
      28	 7.7597661e-01	 1.9485709e-01	 8.1478414e-01	 2.1266447e-01	[ 7.6203450e-01]	 1.7932701e-01


.. parsed-literal::

      29	 8.0910480e-01	 1.9433938e-01	 8.4764753e-01	 2.1107272e-01	[ 8.0354602e-01]	 2.0362353e-01


.. parsed-literal::

      30	 8.4037561e-01	 1.9843080e-01	 8.8186923e-01	 2.1487889e-01	[ 8.2370709e-01]	 2.1182442e-01


.. parsed-literal::

      31	 8.6864366e-01	 1.9606526e-01	 9.0967781e-01	 2.1312333e-01	[ 8.5735711e-01]	 2.0732951e-01
      32	 8.8550660e-01	 1.9104536e-01	 9.2645876e-01	 2.0913538e-01	[ 8.7040878e-01]	 1.8986368e-01


.. parsed-literal::

      33	 8.9875688e-01	 1.8744817e-01	 9.3994121e-01	 2.0580653e-01	[ 8.8027852e-01]	 2.2618985e-01


.. parsed-literal::

      34	 9.1513681e-01	 1.8375032e-01	 9.5698361e-01	 2.0199319e-01	[ 8.9145944e-01]	 2.2592974e-01


.. parsed-literal::

      35	 9.3481954e-01	 1.8242726e-01	 9.7735622e-01	 2.0052048e-01	[ 9.0509556e-01]	 2.2086620e-01
      36	 9.5201027e-01	 1.8091890e-01	 9.9494352e-01	 1.9796312e-01	[ 9.1731285e-01]	 1.9775367e-01


.. parsed-literal::

      37	 9.6334616e-01	 1.8093658e-01	 1.0063158e+00	 1.9783068e-01	[ 9.3707475e-01]	 2.0758319e-01
      38	 9.7840968e-01	 1.7999011e-01	 1.0218729e+00	 1.9591666e-01	[ 9.5704078e-01]	 1.8301678e-01


.. parsed-literal::

      39	 9.9877164e-01	 1.7802426e-01	 1.0437106e+00	 1.9358906e-01	[ 9.8681152e-01]	 2.0630479e-01
      40	 1.0137618e+00	 1.7639925e-01	 1.0595076e+00	 1.9097612e-01	[ 9.9551457e-01]	 1.9574761e-01


.. parsed-literal::

      41	 1.0258516e+00	 1.7454342e-01	 1.0716998e+00	 1.8940175e-01	[ 1.0075496e+00]	 2.2010708e-01


.. parsed-literal::

      42	 1.0424310e+00	 1.7195160e-01	 1.0891912e+00	 1.8676634e-01	[ 1.0199158e+00]	 2.1705675e-01


.. parsed-literal::

      43	 1.0532479e+00	 1.7075094e-01	 1.1001395e+00	 1.8553822e-01	[ 1.0300889e+00]	 2.1894217e-01
      44	 1.0623815e+00	 1.7005581e-01	 1.1092361e+00	 1.8463616e-01	[ 1.0433389e+00]	 1.8579578e-01


.. parsed-literal::

      45	 1.0692937e+00	 1.6936918e-01	 1.1159232e+00	 1.8394760e-01	[ 1.0521298e+00]	 2.0892692e-01


.. parsed-literal::

      46	 1.0799266e+00	 1.6763560e-01	 1.1267140e+00	 1.8268777e-01	[ 1.0646299e+00]	 2.2036338e-01


.. parsed-literal::

      47	 1.0870705e+00	 1.6631819e-01	 1.1341789e+00	 1.8237688e-01	  1.0542120e+00 	 2.1672678e-01


.. parsed-literal::

      48	 1.0987216e+00	 1.6444622e-01	 1.1457513e+00	 1.8077575e-01	[ 1.0691311e+00]	 2.1746850e-01


.. parsed-literal::

      49	 1.1079224e+00	 1.6307287e-01	 1.1551323e+00	 1.7955055e-01	[ 1.0799178e+00]	 2.1707892e-01


.. parsed-literal::

      50	 1.1165315e+00	 1.6177524e-01	 1.1640405e+00	 1.7825445e-01	[ 1.0894243e+00]	 2.1708274e-01


.. parsed-literal::

      51	 1.1327618e+00	 1.5864831e-01	 1.1809094e+00	 1.7555896e-01	[ 1.1069154e+00]	 2.1147728e-01


.. parsed-literal::

      52	 1.1403172e+00	 1.5636862e-01	 1.1892390e+00	 1.7157577e-01	[ 1.1235467e+00]	 2.2186518e-01


.. parsed-literal::

      53	 1.1505956e+00	 1.5575343e-01	 1.1989446e+00	 1.7180791e-01	[ 1.1322313e+00]	 2.1437979e-01


.. parsed-literal::

      54	 1.1586103e+00	 1.5448574e-01	 1.2068559e+00	 1.7111356e-01	[ 1.1384794e+00]	 2.1139288e-01


.. parsed-literal::

      55	 1.1697281e+00	 1.5211095e-01	 1.2181391e+00	 1.6913036e-01	[ 1.1468497e+00]	 2.1269870e-01


.. parsed-literal::

      56	 1.1813469e+00	 1.5044418e-01	 1.2300306e+00	 1.6655811e-01	[ 1.1518062e+00]	 2.1536851e-01


.. parsed-literal::

      57	 1.1923949e+00	 1.4700082e-01	 1.2413688e+00	 1.6369481e-01	[ 1.1588107e+00]	 2.0383072e-01


.. parsed-literal::

      58	 1.2023301e+00	 1.4678848e-01	 1.2512805e+00	 1.6284646e-01	[ 1.1651556e+00]	 2.1187854e-01


.. parsed-literal::

      59	 1.2121186e+00	 1.4586221e-01	 1.2613492e+00	 1.6161070e-01	[ 1.1657869e+00]	 2.1526742e-01


.. parsed-literal::

      60	 1.2237917e+00	 1.4467970e-01	 1.2735792e+00	 1.6028055e-01	[ 1.1715359e+00]	 2.0739555e-01


.. parsed-literal::

      61	 1.2346247e+00	 1.4376464e-01	 1.2843948e+00	 1.5845828e-01	[ 1.1720874e+00]	 2.0998859e-01
      62	 1.2417519e+00	 1.4232277e-01	 1.2915053e+00	 1.5694809e-01	[ 1.1751028e+00]	 1.8649912e-01


.. parsed-literal::

      63	 1.2525389e+00	 1.4043364e-01	 1.3027263e+00	 1.5465501e-01	[ 1.1817474e+00]	 2.0488095e-01


.. parsed-literal::

      64	 1.2595435e+00	 1.3925161e-01	 1.3100740e+00	 1.5310079e-01	[ 1.1824737e+00]	 2.0282316e-01
      65	 1.2651789e+00	 1.3917660e-01	 1.3156576e+00	 1.5271876e-01	[ 1.1926371e+00]	 1.8973804e-01


.. parsed-literal::

      66	 1.2711477e+00	 1.3942039e-01	 1.3217462e+00	 1.5238066e-01	[ 1.2035521e+00]	 1.7954683e-01
      67	 1.2772533e+00	 1.3906764e-01	 1.3279438e+00	 1.5165642e-01	[ 1.2127595e+00]	 1.7621303e-01


.. parsed-literal::

      68	 1.2889828e+00	 1.3882035e-01	 1.3399040e+00	 1.4984863e-01	[ 1.2263983e+00]	 2.0343709e-01
      69	 1.2950911e+00	 1.3767184e-01	 1.3461328e+00	 1.4910139e-01	[ 1.2334058e+00]	 1.8840289e-01


.. parsed-literal::

      70	 1.3014521e+00	 1.3779217e-01	 1.3522501e+00	 1.4888911e-01	[ 1.2368025e+00]	 2.0723510e-01


.. parsed-literal::

      71	 1.3056977e+00	 1.3755472e-01	 1.3565337e+00	 1.4824867e-01	[ 1.2379040e+00]	 2.0272636e-01


.. parsed-literal::

      72	 1.3122299e+00	 1.3718351e-01	 1.3631869e+00	 1.4765753e-01	[ 1.2382706e+00]	 2.0710993e-01


.. parsed-literal::

      73	 1.3185100e+00	 1.3687894e-01	 1.3698329e+00	 1.4657921e-01	  1.2342932e+00 	 2.1667337e-01
      74	 1.3264397e+00	 1.3674676e-01	 1.3777030e+00	 1.4642857e-01	[ 1.2386845e+00]	 1.9409227e-01


.. parsed-literal::

      75	 1.3337237e+00	 1.3655281e-01	 1.3849503e+00	 1.4620665e-01	[ 1.2409074e+00]	 2.0348883e-01


.. parsed-literal::

      76	 1.3407160e+00	 1.3661406e-01	 1.3920646e+00	 1.4605111e-01	  1.2400097e+00 	 2.1212196e-01


.. parsed-literal::

      77	 1.3440272e+00	 1.3655783e-01	 1.3956383e+00	 1.4543385e-01	  1.2169371e+00 	 2.1374440e-01


.. parsed-literal::

      78	 1.3507154e+00	 1.3619542e-01	 1.4020863e+00	 1.4531669e-01	  1.2290501e+00 	 2.1350956e-01


.. parsed-literal::

      79	 1.3546543e+00	 1.3587877e-01	 1.4060822e+00	 1.4500207e-01	  1.2274632e+00 	 2.0081854e-01


.. parsed-literal::

      80	 1.3590107e+00	 1.3551736e-01	 1.4105619e+00	 1.4489239e-01	  1.2192565e+00 	 2.0862889e-01
      81	 1.3656559e+00	 1.3497028e-01	 1.4175224e+00	 1.4461659e-01	  1.1966453e+00 	 1.7683578e-01


.. parsed-literal::

      82	 1.3716222e+00	 1.3469055e-01	 1.4237298e+00	 1.4491536e-01	  1.1708707e+00 	 1.8998361e-01


.. parsed-literal::

      83	 1.3755885e+00	 1.3448565e-01	 1.4276078e+00	 1.4487904e-01	  1.1797365e+00 	 2.0880055e-01


.. parsed-literal::

      84	 1.3800870e+00	 1.3422787e-01	 1.4322253e+00	 1.4474545e-01	  1.1832933e+00 	 2.1397066e-01


.. parsed-literal::

      85	 1.3856663e+00	 1.3384918e-01	 1.4379835e+00	 1.4457716e-01	  1.1860216e+00 	 2.1871114e-01


.. parsed-literal::

      86	 1.3904667e+00	 1.3306922e-01	 1.4432514e+00	 1.4419211e-01	  1.1796389e+00 	 2.0923090e-01


.. parsed-literal::

      87	 1.3965126e+00	 1.3298086e-01	 1.4490733e+00	 1.4415179e-01	  1.1887420e+00 	 2.1160197e-01


.. parsed-literal::

      88	 1.3986799e+00	 1.3274882e-01	 1.4511242e+00	 1.4374487e-01	  1.1939142e+00 	 2.1950912e-01


.. parsed-literal::

      89	 1.4029016e+00	 1.3212436e-01	 1.4553988e+00	 1.4293589e-01	  1.1949263e+00 	 2.1390438e-01


.. parsed-literal::

      90	 1.4077570e+00	 1.3123136e-01	 1.4603797e+00	 1.4133103e-01	  1.2074159e+00 	 2.2051096e-01
      91	 1.4129138e+00	 1.3083765e-01	 1.4656166e+00	 1.4070079e-01	  1.2130862e+00 	 1.8992233e-01


.. parsed-literal::

      92	 1.4181851e+00	 1.3051592e-01	 1.4710955e+00	 1.4024640e-01	  1.2176394e+00 	 1.9471526e-01


.. parsed-literal::

      93	 1.4213466e+00	 1.3034817e-01	 1.4742620e+00	 1.3984063e-01	  1.2162838e+00 	 2.2157812e-01
      94	 1.4256671e+00	 1.3012746e-01	 1.4785835e+00	 1.3939981e-01	  1.2137405e+00 	 1.7244172e-01


.. parsed-literal::

      95	 1.4286208e+00	 1.2984867e-01	 1.4816831e+00	 1.3866048e-01	  1.2002784e+00 	 2.0578265e-01


.. parsed-literal::

      96	 1.4313782e+00	 1.2974295e-01	 1.4843849e+00	 1.3834489e-01	  1.2021841e+00 	 2.1545982e-01


.. parsed-literal::

      97	 1.4334577e+00	 1.2966004e-01	 1.4864690e+00	 1.3808263e-01	  1.2045123e+00 	 2.0921397e-01


.. parsed-literal::

      98	 1.4366150e+00	 1.2948140e-01	 1.4897124e+00	 1.3738290e-01	  1.2051965e+00 	 2.1722865e-01


.. parsed-literal::

      99	 1.4394309e+00	 1.2939987e-01	 1.4926483e+00	 1.3664852e-01	  1.2055604e+00 	 2.1822762e-01
     100	 1.4428188e+00	 1.2915820e-01	 1.4960774e+00	 1.3578188e-01	  1.2083375e+00 	 1.7244077e-01


.. parsed-literal::

     101	 1.4452796e+00	 1.2907535e-01	 1.4985398e+00	 1.3555050e-01	  1.2121017e+00 	 1.9306874e-01
     102	 1.4474992e+00	 1.2903861e-01	 1.5007986e+00	 1.3544398e-01	  1.2146842e+00 	 1.8133688e-01


.. parsed-literal::

     103	 1.4494699e+00	 1.2898727e-01	 1.5028278e+00	 1.3551500e-01	  1.2191079e+00 	 2.1335292e-01


.. parsed-literal::

     104	 1.4512975e+00	 1.2891759e-01	 1.5046752e+00	 1.3577059e-01	  1.2236592e+00 	 2.2093296e-01


.. parsed-literal::

     105	 1.4537257e+00	 1.2888786e-01	 1.5072145e+00	 1.3586511e-01	  1.2304870e+00 	 2.0950031e-01
     106	 1.4559588e+00	 1.2890343e-01	 1.5095158e+00	 1.3593073e-01	  1.2407474e+00 	 1.8554211e-01


.. parsed-literal::

     107	 1.4576291e+00	 1.2887344e-01	 1.5111926e+00	 1.3561527e-01	[ 1.2473361e+00]	 2.1129847e-01
     108	 1.4606324e+00	 1.2864533e-01	 1.5142488e+00	 1.3502939e-01	[ 1.2564048e+00]	 1.8589330e-01


.. parsed-literal::

     109	 1.4622414e+00	 1.2858944e-01	 1.5158955e+00	 1.3460315e-01	[ 1.2684974e+00]	 2.1615648e-01


.. parsed-literal::

     110	 1.4638432e+00	 1.2849391e-01	 1.5174385e+00	 1.3466574e-01	  1.2661077e+00 	 2.1124768e-01
     111	 1.4659932e+00	 1.2840176e-01	 1.5195867e+00	 1.3474789e-01	  1.2646038e+00 	 1.9400167e-01


.. parsed-literal::

     112	 1.4680460e+00	 1.2840107e-01	 1.5216499e+00	 1.3478173e-01	  1.2645454e+00 	 2.0389628e-01


.. parsed-literal::

     113	 1.4691226e+00	 1.2818586e-01	 1.5228416e+00	 1.3462368e-01	  1.2619111e+00 	 2.1622038e-01
     114	 1.4715958e+00	 1.2825447e-01	 1.5252588e+00	 1.3463467e-01	  1.2654655e+00 	 1.9248772e-01


.. parsed-literal::

     115	 1.4728543e+00	 1.2823049e-01	 1.5264993e+00	 1.3465013e-01	  1.2676712e+00 	 2.0342541e-01


.. parsed-literal::

     116	 1.4750055e+00	 1.2803430e-01	 1.5286661e+00	 1.3457214e-01	[ 1.2690517e+00]	 2.1220493e-01


.. parsed-literal::

     117	 1.4761598e+00	 1.2795723e-01	 1.5298428e+00	 1.3469539e-01	  1.2689495e+00 	 3.2642841e-01


.. parsed-literal::

     118	 1.4777633e+00	 1.2775938e-01	 1.5314622e+00	 1.3465555e-01	  1.2681159e+00 	 2.1144414e-01
     119	 1.4797672e+00	 1.2754551e-01	 1.5334905e+00	 1.3463076e-01	  1.2662375e+00 	 1.7995024e-01


.. parsed-literal::

     120	 1.4818071e+00	 1.2739446e-01	 1.5355614e+00	 1.3478799e-01	  1.2642409e+00 	 2.0747852e-01


.. parsed-literal::

     121	 1.4839731e+00	 1.2725127e-01	 1.5377685e+00	 1.3498231e-01	  1.2588785e+00 	 2.1540403e-01
     122	 1.4859233e+00	 1.2722777e-01	 1.5397123e+00	 1.3563375e-01	  1.2578962e+00 	 2.0243955e-01


.. parsed-literal::

     123	 1.4875038e+00	 1.2724197e-01	 1.5412303e+00	 1.3558584e-01	  1.2594957e+00 	 2.0370245e-01


.. parsed-literal::

     124	 1.4892731e+00	 1.2708550e-01	 1.5429943e+00	 1.3537356e-01	  1.2586233e+00 	 2.1920109e-01


.. parsed-literal::

     125	 1.4905297e+00	 1.2706920e-01	 1.5442650e+00	 1.3534599e-01	  1.2580068e+00 	 2.1894598e-01


.. parsed-literal::

     126	 1.4919988e+00	 1.2692697e-01	 1.5457838e+00	 1.3524865e-01	  1.2555166e+00 	 2.1543837e-01


.. parsed-literal::

     127	 1.4939898e+00	 1.2673684e-01	 1.5478601e+00	 1.3510196e-01	  1.2498291e+00 	 2.1862817e-01


.. parsed-literal::

     128	 1.4943790e+00	 1.2656233e-01	 1.5483417e+00	 1.3507411e-01	  1.2388078e+00 	 2.1042514e-01


.. parsed-literal::

     129	 1.4962127e+00	 1.2657911e-01	 1.5500988e+00	 1.3490534e-01	  1.2435463e+00 	 2.1961021e-01


.. parsed-literal::

     130	 1.4969674e+00	 1.2658658e-01	 1.5508219e+00	 1.3484512e-01	  1.2443050e+00 	 2.1403456e-01
     131	 1.4982954e+00	 1.2656001e-01	 1.5521208e+00	 1.3476543e-01	  1.2444343e+00 	 1.9825029e-01


.. parsed-literal::

     132	 1.5004679e+00	 1.2647694e-01	 1.5542828e+00	 1.3475009e-01	  1.2458108e+00 	 2.1054482e-01


.. parsed-literal::

     133	 1.5017951e+00	 1.2627283e-01	 1.5557638e+00	 1.3463218e-01	  1.2363035e+00 	 2.1028924e-01
     134	 1.5046296e+00	 1.2620945e-01	 1.5585905e+00	 1.3493772e-01	  1.2430440e+00 	 2.0160484e-01


.. parsed-literal::

     135	 1.5056965e+00	 1.2614027e-01	 1.5596591e+00	 1.3503585e-01	  1.2435112e+00 	 2.1267843e-01


.. parsed-literal::

     136	 1.5072302e+00	 1.2594420e-01	 1.5612939e+00	 1.3522894e-01	  1.2382557e+00 	 2.0457721e-01


.. parsed-literal::

     137	 1.5081671e+00	 1.2583334e-01	 1.5623651e+00	 1.3550621e-01	  1.2358202e+00 	 2.1048307e-01


.. parsed-literal::

     138	 1.5095516e+00	 1.2571000e-01	 1.5637512e+00	 1.3546594e-01	  1.2341468e+00 	 2.1244359e-01


.. parsed-literal::

     139	 1.5106247e+00	 1.2560512e-01	 1.5648392e+00	 1.3545485e-01	  1.2328487e+00 	 2.1404648e-01


.. parsed-literal::

     140	 1.5117013e+00	 1.2549711e-01	 1.5659232e+00	 1.3545374e-01	  1.2320407e+00 	 2.1265960e-01


.. parsed-literal::

     141	 1.5138566e+00	 1.2520185e-01	 1.5681103e+00	 1.3535049e-01	  1.2301619e+00 	 2.1379924e-01


.. parsed-literal::

     142	 1.5150289e+00	 1.2501720e-01	 1.5693096e+00	 1.3538394e-01	  1.2259903e+00 	 3.2715154e-01


.. parsed-literal::

     143	 1.5163148e+00	 1.2490897e-01	 1.5705662e+00	 1.3533221e-01	  1.2265709e+00 	 2.1178913e-01


.. parsed-literal::

     144	 1.5173850e+00	 1.2482793e-01	 1.5716447e+00	 1.3530998e-01	  1.2256352e+00 	 2.2410560e-01
     145	 1.5190126e+00	 1.2469247e-01	 1.5733432e+00	 1.3540606e-01	  1.2199972e+00 	 1.9326973e-01


.. parsed-literal::

     146	 1.5204661e+00	 1.2463453e-01	 1.5749840e+00	 1.3582776e-01	  1.2116107e+00 	 1.8273258e-01
     147	 1.5216407e+00	 1.2454474e-01	 1.5761789e+00	 1.3579462e-01	  1.2094080e+00 	 2.0398784e-01


.. parsed-literal::

     148	 1.5226440e+00	 1.2450931e-01	 1.5771876e+00	 1.3590116e-01	  1.2075652e+00 	 2.0553303e-01


.. parsed-literal::

     149	 1.5238275e+00	 1.2449755e-01	 1.5784270e+00	 1.3614788e-01	  1.2038130e+00 	 2.1320820e-01


.. parsed-literal::

     150	 1.5249477e+00	 1.2440783e-01	 1.5795975e+00	 1.3653398e-01	  1.1963736e+00 	 2.1544862e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.19 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe221880af0>



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
    CPU times: user 1.86 s, sys: 44 ms, total: 1.9 s
    Wall time: 615 ms


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

