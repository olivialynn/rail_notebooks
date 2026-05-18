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
       1	-3.4164043e-01	 3.1990852e-01	-3.3140692e-01	 3.2248748e-01	[-3.3652014e-01]	 4.5952678e-01


.. parsed-literal::

       2	-2.7233716e-01	 3.0979939e-01	-2.4830918e-01	 3.1370270e-01	[-2.6019696e-01]	 2.2612453e-01


.. parsed-literal::

       3	-2.2943475e-01	 2.8944977e-01	-1.8707895e-01	 2.9487351e-01	[-2.1028797e-01]	 3.0073047e-01
       4	-1.8279491e-01	 2.6532725e-01	-1.3840571e-01	 2.7095214e-01	[-1.7112670e-01]	 1.6698718e-01


.. parsed-literal::

       5	-9.8776922e-02	 2.5494288e-01	-6.7183003e-02	 2.6091186e-01	[-9.7392617e-02]	 2.1212888e-01
       6	-6.3502158e-02	 2.4984542e-01	-3.4080041e-02	 2.5571447e-01	[-5.5130601e-02]	 1.7026067e-01


.. parsed-literal::

       7	-4.5495846e-02	 2.4699348e-01	-2.1161955e-02	 2.5279309e-01	[-4.3524524e-02]	 2.0082903e-01
       8	-3.0930659e-02	 2.4451557e-01	-1.0638686e-02	 2.5074481e-01	[-3.6341689e-02]	 1.9973326e-01


.. parsed-literal::

       9	-1.9215021e-02	 2.4235654e-01	-1.1067381e-03	 2.4895161e-01	[-2.7998697e-02]	 1.9366169e-01


.. parsed-literal::

      10	-1.1719546e-02	 2.4079192e-01	 3.9565573e-03	 2.4731614e-01	[-2.6369651e-02]	 2.1599221e-01
      11	-5.5947340e-03	 2.3991494e-01	 9.8606286e-03	 2.4651745e-01	[-1.7661936e-02]	 2.0174074e-01


.. parsed-literal::

      12	-2.7929590e-03	 2.3939630e-01	 1.2332949e-02	 2.4619514e-01	[-1.6158773e-02]	 1.9275498e-01


.. parsed-literal::

      13	 6.4859740e-04	 2.3875627e-01	 1.5561630e-02	 2.4575741e-01	[-1.3711675e-02]	 2.1660614e-01


.. parsed-literal::

      14	 1.1308412e-01	 2.2642083e-01	 1.3589763e-01	 2.3330512e-01	[ 1.2042822e-01]	 4.3344450e-01
      15	 2.0274987e-01	 2.1750960e-01	 2.2838324e-01	 2.2624809e-01	[ 2.0778249e-01]	 1.9450402e-01


.. parsed-literal::

      16	 2.9195213e-01	 2.1085279e-01	 3.2248173e-01	 2.2093086e-01	[ 2.7391612e-01]	 2.0755553e-01
      17	 3.3735114e-01	 2.0634572e-01	 3.6884335e-01	 2.1781859e-01	[ 3.1849935e-01]	 1.9966316e-01


.. parsed-literal::

      18	 4.0246921e-01	 2.0185044e-01	 4.3440382e-01	 2.1298215e-01	[ 3.9597694e-01]	 2.1271825e-01


.. parsed-literal::

      19	 5.3377474e-01	 1.9916263e-01	 5.6921614e-01	 2.1212663e-01	[ 5.3245606e-01]	 2.0516801e-01


.. parsed-literal::

      20	 5.9803581e-01	 2.0118208e-01	 6.3718868e-01	 2.1524762e-01	[ 6.2105111e-01]	 2.0509887e-01


.. parsed-literal::

      21	 6.4608559e-01	 1.9761534e-01	 6.8433492e-01	 2.0944506e-01	[ 6.7456314e-01]	 2.0825005e-01


.. parsed-literal::

      22	 6.8465892e-01	 1.9486974e-01	 7.2274321e-01	 2.0756689e-01	[ 7.0067305e-01]	 2.2195745e-01
      23	 7.1123674e-01	 1.9325786e-01	 7.4907620e-01	 2.0685466e-01	[ 7.0707567e-01]	 2.0109820e-01


.. parsed-literal::

      24	 7.4061822e-01	 1.9093555e-01	 7.7812588e-01	 2.0319665e-01	[ 7.4590183e-01]	 1.8524671e-01


.. parsed-literal::

      25	 7.6137398e-01	 1.8787117e-01	 8.0018596e-01	 2.0074757e-01	[ 7.7304452e-01]	 2.1919799e-01


.. parsed-literal::

      26	 7.8610699e-01	 1.8861716e-01	 8.2550468e-01	 2.0104175e-01	[ 7.9485231e-01]	 2.1300793e-01


.. parsed-literal::

      27	 8.0886294e-01	 1.8507328e-01	 8.4892329e-01	 1.9880570e-01	[ 8.1507217e-01]	 2.0794296e-01
      28	 8.3582892e-01	 1.8260455e-01	 8.7765211e-01	 1.9698414e-01	[ 8.4078851e-01]	 2.0419979e-01


.. parsed-literal::

      29	 8.5988781e-01	 1.8011851e-01	 9.0266866e-01	 1.9534412e-01	[ 8.6069983e-01]	 2.0115232e-01


.. parsed-literal::

      30	 8.8130831e-01	 1.7748177e-01	 9.2453712e-01	 1.9385579e-01	[ 8.8016929e-01]	 2.1087241e-01
      31	 9.0248456e-01	 1.7625353e-01	 9.4633646e-01	 1.9409769e-01	[ 8.9836245e-01]	 2.0732379e-01


.. parsed-literal::

      32	 9.2663004e-01	 1.7682334e-01	 9.7114390e-01	 1.9578405e-01	[ 9.0962009e-01]	 1.8920279e-01


.. parsed-literal::

      33	 9.4028585e-01	 1.7502306e-01	 9.8526993e-01	 1.9476291e-01	[ 9.2226250e-01]	 2.0605707e-01


.. parsed-literal::

      34	 9.5067120e-01	 1.7394440e-01	 9.9527488e-01	 1.9371060e-01	[ 9.3213156e-01]	 2.0339036e-01


.. parsed-literal::

      35	 9.6145427e-01	 1.7217398e-01	 1.0060223e+00	 1.9212241e-01	[ 9.4217219e-01]	 2.0453954e-01


.. parsed-literal::

      36	 9.7449997e-01	 1.7011924e-01	 1.0192347e+00	 1.9000834e-01	[ 9.5313889e-01]	 2.0255685e-01
      37	 9.9665350e-01	 1.6540257e-01	 1.0431129e+00	 1.8571270e-01	[ 9.6327432e-01]	 1.9216824e-01


.. parsed-literal::

      38	 1.0076435e+00	 1.6257061e-01	 1.0545425e+00	 1.8361231e-01	[ 9.6714486e-01]	 2.0013237e-01


.. parsed-literal::

      39	 1.0157657e+00	 1.6178980e-01	 1.0624850e+00	 1.8318436e-01	[ 9.8013016e-01]	 2.0917439e-01


.. parsed-literal::

      40	 1.0266843e+00	 1.5987389e-01	 1.0737227e+00	 1.8191242e-01	[ 9.9155703e-01]	 2.1195531e-01


.. parsed-literal::

      41	 1.0349263e+00	 1.5792148e-01	 1.0827486e+00	 1.8064358e-01	[ 9.9165242e-01]	 2.0556259e-01


.. parsed-literal::

      42	 1.0436503e+00	 1.5620767e-01	 1.0918612e+00	 1.7934815e-01	[ 9.9899063e-01]	 2.1633506e-01


.. parsed-literal::

      43	 1.0532659e+00	 1.5492352e-01	 1.1016196e+00	 1.7829403e-01	[ 1.0047211e+00]	 2.2142768e-01
      44	 1.0639447e+00	 1.5411254e-01	 1.1123413e+00	 1.7758384e-01	[ 1.0181263e+00]	 1.8501735e-01


.. parsed-literal::

      45	 1.0753705e+00	 1.5286900e-01	 1.1239883e+00	 1.7624747e-01	[ 1.0311148e+00]	 1.9794035e-01


.. parsed-literal::

      46	 1.0838570e+00	 1.5232433e-01	 1.1323704e+00	 1.7572640e-01	[ 1.0470691e+00]	 2.0760536e-01
      47	 1.0920149e+00	 1.5110553e-01	 1.1405675e+00	 1.7507537e-01	[ 1.0550904e+00]	 1.9727969e-01


.. parsed-literal::

      48	 1.1036412e+00	 1.4912805e-01	 1.1524781e+00	 1.7418320e-01	[ 1.0614887e+00]	 1.8392134e-01


.. parsed-literal::

      49	 1.1100365e+00	 1.4771738e-01	 1.1591787e+00	 1.7420718e-01	[ 1.0670169e+00]	 2.0694518e-01


.. parsed-literal::

      50	 1.1169645e+00	 1.4702958e-01	 1.1660105e+00	 1.7366246e-01	[ 1.0728202e+00]	 2.1295857e-01
      51	 1.1254835e+00	 1.4632110e-01	 1.1745885e+00	 1.7325827e-01	[ 1.0790439e+00]	 1.9560862e-01


.. parsed-literal::

      52	 1.1339044e+00	 1.4527839e-01	 1.1832249e+00	 1.7239492e-01	[ 1.0864308e+00]	 2.3402762e-01


.. parsed-literal::

      53	 1.1421205e+00	 1.4454799e-01	 1.1922002e+00	 1.7241913e-01	  1.0863373e+00 	 2.0930052e-01
      54	 1.1502718e+00	 1.4418234e-01	 1.2004429e+00	 1.7227267e-01	[ 1.0915839e+00]	 1.9978690e-01


.. parsed-literal::

      55	 1.1575262e+00	 1.4391041e-01	 1.2078908e+00	 1.7214985e-01	[ 1.0985035e+00]	 1.8024755e-01


.. parsed-literal::

      56	 1.1667372e+00	 1.4349304e-01	 1.2175071e+00	 1.7225446e-01	[ 1.1008260e+00]	 2.1324420e-01


.. parsed-literal::

      57	 1.1720853e+00	 1.4381773e-01	 1.2233975e+00	 1.7340660e-01	[ 1.1027634e+00]	 2.0410776e-01


.. parsed-literal::

      58	 1.1799593e+00	 1.4272591e-01	 1.2309500e+00	 1.7223462e-01	[ 1.1117105e+00]	 2.1006584e-01
      59	 1.1834397e+00	 1.4239964e-01	 1.2344200e+00	 1.7200612e-01	[ 1.1142607e+00]	 1.8563199e-01


.. parsed-literal::

      60	 1.1930478e+00	 1.4161828e-01	 1.2444205e+00	 1.7163157e-01	[ 1.1171856e+00]	 1.8760514e-01


.. parsed-literal::

      61	 1.1996716e+00	 1.4103115e-01	 1.2514189e+00	 1.7115551e-01	  1.1171287e+00 	 2.0419073e-01


.. parsed-literal::

      62	 1.2089927e+00	 1.4065152e-01	 1.2611568e+00	 1.7105882e-01	[ 1.1227776e+00]	 2.0757508e-01


.. parsed-literal::

      63	 1.2172598e+00	 1.4025670e-01	 1.2695803e+00	 1.7052680e-01	[ 1.1309045e+00]	 2.1459723e-01


.. parsed-literal::

      64	 1.2244476e+00	 1.3953897e-01	 1.2769325e+00	 1.7003602e-01	[ 1.1355043e+00]	 2.0800018e-01


.. parsed-literal::

      65	 1.2326806e+00	 1.3945169e-01	 1.2853536e+00	 1.6964084e-01	[ 1.1382783e+00]	 2.0547581e-01


.. parsed-literal::

      66	 1.2398094e+00	 1.3939434e-01	 1.2926798e+00	 1.6986803e-01	[ 1.1407150e+00]	 2.0879793e-01


.. parsed-literal::

      67	 1.2461125e+00	 1.3915826e-01	 1.2990495e+00	 1.6961561e-01	[ 1.1413034e+00]	 2.0982361e-01


.. parsed-literal::

      68	 1.2551717e+00	 1.3886476e-01	 1.3085440e+00	 1.6948356e-01	  1.1388477e+00 	 2.1037364e-01


.. parsed-literal::

      69	 1.2572556e+00	 1.3856244e-01	 1.3109729e+00	 1.6915374e-01	  1.1323755e+00 	 2.1163893e-01


.. parsed-literal::

      70	 1.2656853e+00	 1.3834485e-01	 1.3191493e+00	 1.6907853e-01	[ 1.1468311e+00]	 2.0359564e-01


.. parsed-literal::

      71	 1.2696861e+00	 1.3812130e-01	 1.3231746e+00	 1.6898006e-01	[ 1.1554775e+00]	 2.1186042e-01
      72	 1.2745261e+00	 1.3778913e-01	 1.3281229e+00	 1.6877735e-01	[ 1.1645103e+00]	 1.9663072e-01


.. parsed-literal::

      73	 1.2785301e+00	 1.3748141e-01	 1.3325914e+00	 1.6831078e-01	[ 1.1719244e+00]	 1.9852185e-01


.. parsed-literal::

      74	 1.2850108e+00	 1.3717877e-01	 1.3390435e+00	 1.6820641e-01	[ 1.1771370e+00]	 2.0673585e-01


.. parsed-literal::

      75	 1.2882363e+00	 1.3696497e-01	 1.3423224e+00	 1.6804683e-01	[ 1.1782666e+00]	 2.0147538e-01
      76	 1.2930808e+00	 1.3648951e-01	 1.3472785e+00	 1.6766850e-01	[ 1.1800653e+00]	 1.9240546e-01


.. parsed-literal::

      77	 1.3005527e+00	 1.3585639e-01	 1.3548880e+00	 1.6717677e-01	[ 1.1869668e+00]	 1.8939853e-01


.. parsed-literal::

      78	 1.3049489e+00	 1.3496699e-01	 1.3599506e+00	 1.6664227e-01	[ 1.1917929e+00]	 2.0618796e-01
      79	 1.3166092e+00	 1.3475253e-01	 1.3712221e+00	 1.6617876e-01	[ 1.2070706e+00]	 1.7621112e-01


.. parsed-literal::

      80	 1.3203080e+00	 1.3468670e-01	 1.3747313e+00	 1.6616817e-01	[ 1.2127920e+00]	 2.0136094e-01
      81	 1.3258521e+00	 1.3448903e-01	 1.3803127e+00	 1.6614528e-01	[ 1.2187590e+00]	 1.9916797e-01


.. parsed-literal::

      82	 1.3318041e+00	 1.3382364e-01	 1.3864231e+00	 1.6592581e-01	[ 1.2232039e+00]	 1.7989326e-01
      83	 1.3369702e+00	 1.3356260e-01	 1.3917559e+00	 1.6585587e-01	[ 1.2242722e+00]	 1.8493891e-01


.. parsed-literal::

      84	 1.3412802e+00	 1.3300081e-01	 1.3961531e+00	 1.6554841e-01	[ 1.2249922e+00]	 1.7213464e-01


.. parsed-literal::

      85	 1.3463334e+00	 1.3246150e-01	 1.4014219e+00	 1.6529571e-01	[ 1.2267174e+00]	 2.0987582e-01


.. parsed-literal::

      86	 1.3516336e+00	 1.3169419e-01	 1.4069974e+00	 1.6497617e-01	[ 1.2274236e+00]	 2.0624304e-01
      87	 1.3571163e+00	 1.3125082e-01	 1.4126562e+00	 1.6502498e-01	[ 1.2340571e+00]	 1.9813824e-01


.. parsed-literal::

      88	 1.3603207e+00	 1.3103476e-01	 1.4158950e+00	 1.6518189e-01	[ 1.2346520e+00]	 2.0615625e-01


.. parsed-literal::

      89	 1.3631936e+00	 1.3088732e-01	 1.4187924e+00	 1.6498190e-01	[ 1.2397241e+00]	 2.0566511e-01


.. parsed-literal::

      90	 1.3679626e+00	 1.3064585e-01	 1.4236175e+00	 1.6482353e-01	[ 1.2429868e+00]	 2.1162820e-01
      91	 1.3729283e+00	 1.3026613e-01	 1.4289290e+00	 1.6432655e-01	[ 1.2458817e+00]	 1.9401050e-01


.. parsed-literal::

      92	 1.3770122e+00	 1.2988385e-01	 1.4331034e+00	 1.6396115e-01	[ 1.2464335e+00]	 2.0636463e-01
      93	 1.3798336e+00	 1.2953821e-01	 1.4359088e+00	 1.6379423e-01	  1.2461053e+00 	 1.8578792e-01


.. parsed-literal::

      94	 1.3835208e+00	 1.2906934e-01	 1.4397707e+00	 1.6356844e-01	[ 1.2468626e+00]	 2.0558810e-01


.. parsed-literal::

      95	 1.3879287e+00	 1.2860750e-01	 1.4444928e+00	 1.6336132e-01	[ 1.2469354e+00]	 2.0289946e-01


.. parsed-literal::

      96	 1.3921415e+00	 1.2818992e-01	 1.4488415e+00	 1.6325377e-01	[ 1.2534527e+00]	 2.0677400e-01
      97	 1.3951417e+00	 1.2808622e-01	 1.4517461e+00	 1.6314959e-01	[ 1.2588026e+00]	 2.0240402e-01


.. parsed-literal::

      98	 1.3990718e+00	 1.2781332e-01	 1.4555795e+00	 1.6288248e-01	[ 1.2644636e+00]	 1.9070935e-01


.. parsed-literal::

      99	 1.4024289e+00	 1.2726528e-01	 1.4590575e+00	 1.6233222e-01	[ 1.2719115e+00]	 2.0681024e-01


.. parsed-literal::

     100	 1.4067036e+00	 1.2690503e-01	 1.4632399e+00	 1.6202889e-01	[ 1.2724607e+00]	 2.1843171e-01


.. parsed-literal::

     101	 1.4093538e+00	 1.2656289e-01	 1.4660241e+00	 1.6168736e-01	  1.2716942e+00 	 2.0867252e-01
     102	 1.4113370e+00	 1.2635329e-01	 1.4680789e+00	 1.6150779e-01	[ 1.2726018e+00]	 1.9382334e-01


.. parsed-literal::

     103	 1.4154040e+00	 1.2587964e-01	 1.4723669e+00	 1.6132751e-01	  1.2709102e+00 	 2.0396852e-01
     104	 1.4184051e+00	 1.2555636e-01	 1.4756012e+00	 1.6126795e-01	  1.2702012e+00 	 1.9436002e-01


.. parsed-literal::

     105	 1.4223784e+00	 1.2546689e-01	 1.4793641e+00	 1.6138015e-01	  1.2723445e+00 	 1.8926573e-01


.. parsed-literal::

     106	 1.4248850e+00	 1.2543548e-01	 1.4817693e+00	 1.6157657e-01	[ 1.2727305e+00]	 2.0680141e-01


.. parsed-literal::

     107	 1.4278742e+00	 1.2533755e-01	 1.4846924e+00	 1.6171707e-01	[ 1.2735751e+00]	 2.0357537e-01
     108	 1.4303236e+00	 1.2501092e-01	 1.4872861e+00	 1.6208245e-01	[ 1.2740193e+00]	 1.9584751e-01


.. parsed-literal::

     109	 1.4349517e+00	 1.2473694e-01	 1.4918167e+00	 1.6179353e-01	[ 1.2777830e+00]	 1.7739224e-01


.. parsed-literal::

     110	 1.4370444e+00	 1.2454726e-01	 1.4939322e+00	 1.6155312e-01	[ 1.2800184e+00]	 2.0689654e-01


.. parsed-literal::

     111	 1.4401193e+00	 1.2417698e-01	 1.4971590e+00	 1.6134506e-01	  1.2798380e+00 	 2.0636225e-01


.. parsed-literal::

     112	 1.4434086e+00	 1.2361434e-01	 1.5006183e+00	 1.6096695e-01	  1.2758156e+00 	 2.0518398e-01


.. parsed-literal::

     113	 1.4469102e+00	 1.2318654e-01	 1.5041476e+00	 1.6093650e-01	  1.2750957e+00 	 2.1919656e-01


.. parsed-literal::

     114	 1.4497436e+00	 1.2293709e-01	 1.5069346e+00	 1.6101388e-01	  1.2723003e+00 	 2.1368384e-01


.. parsed-literal::

     115	 1.4518428e+00	 1.2273648e-01	 1.5089927e+00	 1.6110011e-01	  1.2728251e+00 	 2.1410465e-01
     116	 1.4541199e+00	 1.2247892e-01	 1.5112561e+00	 1.6100497e-01	  1.2714124e+00 	 1.9380236e-01


.. parsed-literal::

     117	 1.4562023e+00	 1.2222554e-01	 1.5133480e+00	 1.6081609e-01	  1.2720086e+00 	 2.1443868e-01
     118	 1.4596064e+00	 1.2163654e-01	 1.5169297e+00	 1.6021864e-01	  1.2705601e+00 	 1.8833947e-01


.. parsed-literal::

     119	 1.4624230e+00	 1.2118625e-01	 1.5200324e+00	 1.5968137e-01	  1.2641151e+00 	 1.9991541e-01
     120	 1.4652955e+00	 1.2083986e-01	 1.5230241e+00	 1.5917104e-01	  1.2620318e+00 	 1.9597006e-01


.. parsed-literal::

     121	 1.4683118e+00	 1.2049881e-01	 1.5261757e+00	 1.5879087e-01	  1.2577311e+00 	 2.0497417e-01


.. parsed-literal::

     122	 1.4703097e+00	 1.2028209e-01	 1.5282929e+00	 1.5842749e-01	  1.2528514e+00 	 2.0750999e-01
     123	 1.4726660e+00	 1.2020000e-01	 1.5306106e+00	 1.5844136e-01	  1.2530140e+00 	 1.6930723e-01


.. parsed-literal::

     124	 1.4755376e+00	 1.2000872e-01	 1.5334307e+00	 1.5841861e-01	  1.2556266e+00 	 1.7217731e-01
     125	 1.4778022e+00	 1.1976222e-01	 1.5356759e+00	 1.5815449e-01	  1.2558502e+00 	 1.9985890e-01


.. parsed-literal::

     126	 1.4807474e+00	 1.1930092e-01	 1.5386389e+00	 1.5769495e-01	  1.2615862e+00 	 1.9928670e-01
     127	 1.4830427e+00	 1.1906813e-01	 1.5409785e+00	 1.5734483e-01	  1.2617470e+00 	 1.9683552e-01


.. parsed-literal::

     128	 1.4849699e+00	 1.1888197e-01	 1.5429440e+00	 1.5695047e-01	  1.2635828e+00 	 1.9695115e-01
     129	 1.4865449e+00	 1.1878405e-01	 1.5445752e+00	 1.5683567e-01	  1.2624925e+00 	 1.9509864e-01


.. parsed-literal::

     130	 1.4882550e+00	 1.1865256e-01	 1.5463241e+00	 1.5679413e-01	  1.2608760e+00 	 1.9779801e-01
     131	 1.4908098e+00	 1.1839865e-01	 1.5488768e+00	 1.5671919e-01	  1.2601512e+00 	 2.0078254e-01


.. parsed-literal::

     132	 1.4921971e+00	 1.1815740e-01	 1.5503424e+00	 1.5679824e-01	  1.2563563e+00 	 1.9609237e-01
     133	 1.4945101e+00	 1.1804471e-01	 1.5524913e+00	 1.5665486e-01	  1.2587450e+00 	 1.9465613e-01


.. parsed-literal::

     134	 1.4961356e+00	 1.1789568e-01	 1.5540249e+00	 1.5649485e-01	  1.2597524e+00 	 1.9818974e-01
     135	 1.4977948e+00	 1.1768681e-01	 1.5556616e+00	 1.5631173e-01	  1.2589203e+00 	 1.9585586e-01


.. parsed-literal::

     136	 1.4998461e+00	 1.1734718e-01	 1.5578235e+00	 1.5609994e-01	  1.2526205e+00 	 2.0883012e-01


.. parsed-literal::

     137	 1.5020077e+00	 1.1708552e-01	 1.5600655e+00	 1.5595206e-01	  1.2488183e+00 	 2.1405435e-01


.. parsed-literal::

     138	 1.5030519e+00	 1.1705448e-01	 1.5611431e+00	 1.5598307e-01	  1.2485263e+00 	 2.1908402e-01


.. parsed-literal::

     139	 1.5046650e+00	 1.1693193e-01	 1.5628637e+00	 1.5601957e-01	  1.2430284e+00 	 2.1955085e-01


.. parsed-literal::

     140	 1.5055574e+00	 1.1690841e-01	 1.5639188e+00	 1.5618373e-01	  1.2428064e+00 	 2.1404266e-01


.. parsed-literal::

     141	 1.5073780e+00	 1.1685232e-01	 1.5656433e+00	 1.5616867e-01	  1.2408986e+00 	 2.1494317e-01
     142	 1.5093246e+00	 1.1675818e-01	 1.5675037e+00	 1.5617974e-01	  1.2378336e+00 	 1.9720745e-01


.. parsed-literal::

     143	 1.5105999e+00	 1.1669380e-01	 1.5687458e+00	 1.5622423e-01	  1.2370428e+00 	 1.8052459e-01
     144	 1.5134522e+00	 1.1645699e-01	 1.5715964e+00	 1.5623205e-01	  1.2347378e+00 	 1.9975448e-01


.. parsed-literal::

     145	 1.5146512e+00	 1.1626074e-01	 1.5728228e+00	 1.5622931e-01	  1.2353372e+00 	 3.1649542e-01
     146	 1.5165595e+00	 1.1607790e-01	 1.5747642e+00	 1.5619766e-01	  1.2349681e+00 	 2.0093584e-01


.. parsed-literal::

     147	 1.5180403e+00	 1.1592544e-01	 1.5762980e+00	 1.5604907e-01	  1.2332342e+00 	 2.0130277e-01
     148	 1.5198580e+00	 1.1570090e-01	 1.5782765e+00	 1.5579988e-01	  1.2357225e+00 	 2.0010781e-01


.. parsed-literal::

     149	 1.5213341e+00	 1.1553092e-01	 1.5798519e+00	 1.5547274e-01	  1.2278345e+00 	 2.0337701e-01
     150	 1.5224996e+00	 1.1551562e-01	 1.5809846e+00	 1.5538461e-01	  1.2294678e+00 	 2.0275736e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.04 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f43d0a5bac0>



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
    CPU times: user 989 ms, sys: 42.9 ms, total: 1.03 s
    Wall time: 391 ms


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

