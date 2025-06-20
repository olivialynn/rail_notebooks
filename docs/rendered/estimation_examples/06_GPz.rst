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
       1	-3.4832998e-01	 3.2224148e-01	-3.3871404e-01	 3.1408559e-01	[-3.2267661e-01]	 4.6631312e-01


.. parsed-literal::

       2	-2.8061498e-01	 3.1287361e-01	-2.5764158e-01	 3.0329769e-01	[-2.2845172e-01]	 2.3432040e-01


.. parsed-literal::

       3	-2.3381182e-01	 2.9104200e-01	-1.8849641e-01	 2.8439925e-01	[-1.6134137e-01]	 3.0567026e-01


.. parsed-literal::

       4	-2.0341059e-01	 2.7554606e-01	-1.5333147e-01	 2.7688113e-01	[-1.5125468e-01]	 2.9925370e-01


.. parsed-literal::

       5	-1.3979950e-01	 2.5952199e-01	-9.6805406e-02	 2.6215445e-01	[-1.0385375e-01]	 2.1069479e-01


.. parsed-literal::

       6	-7.7507269e-02	 2.5299993e-01	-4.6228797e-02	 2.5530025e-01	[-5.2148136e-02]	 2.0245767e-01
       7	-5.8782915e-02	 2.5035998e-01	-3.3743563e-02	 2.4991745e-01	[-3.1430497e-02]	 1.7585182e-01


.. parsed-literal::

       8	-4.1743875e-02	 2.4736821e-01	-2.2286186e-02	 2.4683845e-01	[-1.9040760e-02]	 2.1128941e-01


.. parsed-literal::

       9	-3.3198666e-02	 2.4586041e-01	-1.5906896e-02	 2.4435567e-01	[-9.3612880e-03]	 2.1458101e-01


.. parsed-literal::

      10	-2.1835938e-02	 2.4360856e-01	-6.4177799e-03	 2.4124257e-01	[ 3.8926923e-03]	 2.1774220e-01


.. parsed-literal::

      11	-1.2218349e-02	 2.4199294e-01	 1.7178017e-03	 2.3990840e-01	[ 9.7968580e-03]	 2.1025658e-01


.. parsed-literal::

      12	-7.7691695e-03	 2.4114283e-01	 6.0823647e-03	 2.4008029e-01	[ 1.1151763e-02]	 2.0789385e-01


.. parsed-literal::

      13	-2.2463128e-03	 2.4008137e-01	 1.1949448e-02	 2.4020239e-01	[ 1.2868232e-02]	 2.1157861e-01


.. parsed-literal::

      14	 1.3714419e-01	 2.2642518e-01	 1.6026673e-01	 2.3234403e-01	[ 1.4157622e-01]	 3.1025267e-01


.. parsed-literal::

      15	 1.8639736e-01	 2.2254781e-01	 2.1118226e-01	 2.2544008e-01	[ 1.9621494e-01]	 4.0121698e-01


.. parsed-literal::

      16	 2.2267334e-01	 2.1791312e-01	 2.4892200e-01	 2.2444195e-01	[ 2.2517257e-01]	 2.0555377e-01


.. parsed-literal::

      17	 2.8483117e-01	 2.1324693e-01	 3.1398126e-01	 2.2011532e-01	[ 2.7930082e-01]	 2.0619082e-01


.. parsed-literal::

      18	 3.3760448e-01	 2.0840858e-01	 3.7025368e-01	 2.1565285e-01	[ 3.1667500e-01]	 2.0410442e-01


.. parsed-literal::

      19	 3.9583274e-01	 2.0490236e-01	 4.2979987e-01	 2.1150247e-01	[ 3.7457531e-01]	 2.0931101e-01
      20	 4.6674590e-01	 2.0417754e-01	 5.0076462e-01	 2.1068826e-01	[ 4.4791732e-01]	 1.9985652e-01


.. parsed-literal::

      21	 5.4783240e-01	 2.0015242e-01	 5.8259059e-01	 2.0475482e-01	[ 5.3014258e-01]	 2.0593596e-01


.. parsed-literal::

      22	 6.3079758e-01	 1.9866151e-01	 6.6694230e-01	 2.0063483e-01	[ 6.1692687e-01]	 2.0772529e-01


.. parsed-literal::

      23	 6.6143616e-01	 1.9690600e-01	 7.0068634e-01	 1.9811391e-01	[ 6.4481763e-01]	 2.1050525e-01
      24	 6.9846026e-01	 1.9127107e-01	 7.3615362e-01	 1.9259981e-01	[ 6.8741338e-01]	 1.8594217e-01


.. parsed-literal::

      25	 7.2174594e-01	 1.9107307e-01	 7.5907788e-01	 1.9166865e-01	[ 7.0963031e-01]	 1.9273448e-01


.. parsed-literal::

      26	 7.4754432e-01	 1.9629575e-01	 7.8438472e-01	 1.9552256e-01	[ 7.3264939e-01]	 2.1874976e-01


.. parsed-literal::

      27	 7.5741015e-01	 2.0239229e-01	 7.9476475e-01	 1.9803285e-01	  7.3241425e-01 	 2.1771622e-01
      28	 7.7846813e-01	 1.9530430e-01	 8.1658812e-01	 1.9350527e-01	[ 7.5695340e-01]	 1.8731260e-01


.. parsed-literal::

      29	 7.9567606e-01	 1.9290136e-01	 8.3425903e-01	 1.9252756e-01	[ 7.7494174e-01]	 2.1288848e-01


.. parsed-literal::

      30	 8.1693771e-01	 1.9067993e-01	 8.5623878e-01	 1.9073907e-01	[ 7.9343008e-01]	 2.0884633e-01


.. parsed-literal::

      31	 8.4174562e-01	 1.8996478e-01	 8.8147523e-01	 1.9007530e-01	[ 8.0943723e-01]	 2.1048069e-01


.. parsed-literal::

      32	 8.5722092e-01	 1.8955414e-01	 8.9738112e-01	 1.8830959e-01	[ 8.2625654e-01]	 2.1081901e-01
      33	 8.7411975e-01	 1.8666158e-01	 9.1391665e-01	 1.8565179e-01	[ 8.4445184e-01]	 1.9987893e-01


.. parsed-literal::

      34	 8.9314396e-01	 1.8404132e-01	 9.3286592e-01	 1.8207410e-01	[ 8.6507256e-01]	 2.0546079e-01


.. parsed-literal::

      35	 9.1328818e-01	 1.8057702e-01	 9.5348216e-01	 1.7760186e-01	[ 8.8394521e-01]	 2.1117926e-01


.. parsed-literal::

      36	 9.1952756e-01	 1.8271528e-01	 9.6098353e-01	 1.7572345e-01	  8.8177295e-01 	 2.0342422e-01


.. parsed-literal::

      37	 9.4180647e-01	 1.7950131e-01	 9.8328663e-01	 1.7450344e-01	[ 8.9707470e-01]	 2.1280837e-01


.. parsed-literal::

      38	 9.4971607e-01	 1.7844249e-01	 9.9140196e-01	 1.7412771e-01	[ 9.0422391e-01]	 2.0416069e-01
      39	 9.6216449e-01	 1.7792085e-01	 1.0044620e+00	 1.7402784e-01	[ 9.1110969e-01]	 2.0228553e-01


.. parsed-literal::

      40	 9.7911863e-01	 1.7790744e-01	 1.0223778e+00	 1.7364002e-01	[ 9.2030445e-01]	 2.0072818e-01


.. parsed-literal::

      41	 9.9670640e-01	 1.7778499e-01	 1.0406906e+00	 1.7219493e-01	[ 9.3594760e-01]	 2.0523429e-01


.. parsed-literal::

      42	 1.0066911e+00	 1.7769173e-01	 1.0509133e+00	 1.7126857e-01	[ 9.4449654e-01]	 2.0855594e-01
      43	 1.0266000e+00	 1.7673179e-01	 1.0712445e+00	 1.6921836e-01	[ 9.6421785e-01]	 2.0647311e-01


.. parsed-literal::

      44	 1.0395175e+00	 1.7469077e-01	 1.0851041e+00	 1.6888258e-01	  9.6404097e-01 	 2.0692921e-01
      45	 1.0534800e+00	 1.7538004e-01	 1.0990658e+00	 1.6775966e-01	[ 9.7489668e-01]	 1.8390989e-01


.. parsed-literal::

      46	 1.0611435e+00	 1.7507072e-01	 1.1068622e+00	 1.6751858e-01	[ 9.7823860e-01]	 2.1109653e-01


.. parsed-literal::

      47	 1.0740528e+00	 1.7384316e-01	 1.1201816e+00	 1.6615872e-01	[ 9.8531786e-01]	 2.1894169e-01


.. parsed-literal::

      48	 1.0854568e+00	 1.7346725e-01	 1.1317416e+00	 1.6572288e-01	[ 9.9365168e-01]	 2.1495318e-01


.. parsed-literal::

      49	 1.0964218e+00	 1.7220085e-01	 1.1424550e+00	 1.6510277e-01	[ 1.0049403e+00]	 2.0531368e-01


.. parsed-literal::

      50	 1.1046301e+00	 1.7160866e-01	 1.1505859e+00	 1.6411887e-01	[ 1.0147420e+00]	 2.0591497e-01


.. parsed-literal::

      51	 1.1142624e+00	 1.6979943e-01	 1.1604641e+00	 1.6268865e-01	[ 1.0234593e+00]	 2.1337605e-01


.. parsed-literal::

      52	 1.1232116e+00	 1.6949386e-01	 1.1702409e+00	 1.6165651e-01	  1.0200577e+00 	 2.0479584e-01


.. parsed-literal::

      53	 1.1322564e+00	 1.6845206e-01	 1.1795344e+00	 1.6081307e-01	[ 1.0275550e+00]	 2.0742583e-01


.. parsed-literal::

      54	 1.1420232e+00	 1.6732993e-01	 1.1896830e+00	 1.5975078e-01	[ 1.0332520e+00]	 2.0724821e-01


.. parsed-literal::

      55	 1.1494505e+00	 1.6681354e-01	 1.1972750e+00	 1.5929795e-01	[ 1.0375706e+00]	 2.0993114e-01
      56	 1.1616708e+00	 1.6527788e-01	 1.2097663e+00	 1.5730058e-01	[ 1.0499927e+00]	 1.7383718e-01


.. parsed-literal::

      57	 1.1690331e+00	 1.6388113e-01	 1.2174376e+00	 1.5763679e-01	  1.0488261e+00 	 1.7280698e-01
      58	 1.1764017e+00	 1.6312822e-01	 1.2248405e+00	 1.5700948e-01	[ 1.0553780e+00]	 1.9898415e-01


.. parsed-literal::

      59	 1.1859888e+00	 1.6203930e-01	 1.2348637e+00	 1.5605066e-01	[ 1.0583892e+00]	 2.0762539e-01


.. parsed-literal::

      60	 1.1928131e+00	 1.6125445e-01	 1.2419796e+00	 1.5566910e-01	[ 1.0606612e+00]	 2.1205068e-01


.. parsed-literal::

      61	 1.2007740e+00	 1.6047373e-01	 1.2504518e+00	 1.5475649e-01	[ 1.0665604e+00]	 2.1621490e-01


.. parsed-literal::

      62	 1.2076823e+00	 1.5960528e-01	 1.2573787e+00	 1.5443622e-01	[ 1.0716063e+00]	 2.1543932e-01


.. parsed-literal::

      63	 1.2123798e+00	 1.5945608e-01	 1.2617659e+00	 1.5382818e-01	[ 1.0828796e+00]	 2.0543480e-01


.. parsed-literal::

      64	 1.2192852e+00	 1.5838041e-01	 1.2687073e+00	 1.5296146e-01	[ 1.0944052e+00]	 2.0483232e-01


.. parsed-literal::

      65	 1.2266977e+00	 1.5775158e-01	 1.2763503e+00	 1.5140770e-01	[ 1.1060619e+00]	 2.1840954e-01


.. parsed-literal::

      66	 1.2304804e+00	 1.5685368e-01	 1.2803088e+00	 1.5015237e-01	[ 1.1161055e+00]	 3.0421686e-01


.. parsed-literal::

      67	 1.2361880e+00	 1.5619050e-01	 1.2861160e+00	 1.4946677e-01	[ 1.1198033e+00]	 2.0625949e-01


.. parsed-literal::

      68	 1.2435960e+00	 1.5501422e-01	 1.2936854e+00	 1.4917298e-01	[ 1.1240315e+00]	 2.1024513e-01
      69	 1.2504361e+00	 1.5423839e-01	 1.3005674e+00	 1.4867260e-01	[ 1.1285442e+00]	 1.9096184e-01


.. parsed-literal::

      70	 1.2581872e+00	 1.5311459e-01	 1.3084556e+00	 1.4965375e-01	[ 1.1364678e+00]	 2.0901346e-01


.. parsed-literal::

      71	 1.2650508e+00	 1.5317607e-01	 1.3151372e+00	 1.5000273e-01	[ 1.1466998e+00]	 2.1194053e-01


.. parsed-literal::

      72	 1.2719418e+00	 1.5295990e-01	 1.3220131e+00	 1.4966804e-01	[ 1.1549524e+00]	 2.0865822e-01


.. parsed-literal::

      73	 1.2791867e+00	 1.5263839e-01	 1.3295097e+00	 1.4941207e-01	[ 1.1646057e+00]	 2.1977448e-01


.. parsed-literal::

      74	 1.2871808e+00	 1.5222974e-01	 1.3379680e+00	 1.4909892e-01	[ 1.1740326e+00]	 2.1418548e-01
      75	 1.2940913e+00	 1.5126394e-01	 1.3450056e+00	 1.4891929e-01	[ 1.1773376e+00]	 2.0105004e-01


.. parsed-literal::

      76	 1.3000022e+00	 1.5054621e-01	 1.3508690e+00	 1.4889651e-01	[ 1.1777699e+00]	 2.0604849e-01


.. parsed-literal::

      77	 1.3063454e+00	 1.4922918e-01	 1.3574462e+00	 1.4818530e-01	  1.1776651e+00 	 2.0202851e-01


.. parsed-literal::

      78	 1.3091053e+00	 1.4957298e-01	 1.3604175e+00	 1.4868061e-01	  1.1644931e+00 	 2.1313405e-01


.. parsed-literal::

      79	 1.3154576e+00	 1.4917308e-01	 1.3665722e+00	 1.4802476e-01	  1.1768169e+00 	 2.0966983e-01


.. parsed-literal::

      80	 1.3190744e+00	 1.4900135e-01	 1.3702629e+00	 1.4769856e-01	[ 1.1827042e+00]	 2.1364069e-01
      81	 1.3236893e+00	 1.4873855e-01	 1.3749605e+00	 1.4735968e-01	[ 1.1874549e+00]	 1.9818234e-01


.. parsed-literal::

      82	 1.3312132e+00	 1.4780472e-01	 1.3827018e+00	 1.4677350e-01	[ 1.1934093e+00]	 1.8858242e-01


.. parsed-literal::

      83	 1.3338391e+00	 1.4785176e-01	 1.3856604e+00	 1.4621369e-01	[ 1.1986007e+00]	 2.1819353e-01
      84	 1.3403952e+00	 1.4739610e-01	 1.3919408e+00	 1.4621109e-01	[ 1.2043174e+00]	 1.9625854e-01


.. parsed-literal::

      85	 1.3439865e+00	 1.4713305e-01	 1.3954642e+00	 1.4602856e-01	[ 1.2074733e+00]	 2.0695901e-01
      86	 1.3499679e+00	 1.4696847e-01	 1.4015268e+00	 1.4558591e-01	[ 1.2113723e+00]	 1.9031215e-01


.. parsed-literal::

      87	 1.3567691e+00	 1.4688057e-01	 1.4083672e+00	 1.4522528e-01	[ 1.2139588e+00]	 2.1141148e-01


.. parsed-literal::

      88	 1.3613488e+00	 1.4788729e-01	 1.4131990e+00	 1.4556813e-01	  1.2097161e+00 	 2.0426559e-01


.. parsed-literal::

      89	 1.3663340e+00	 1.4731906e-01	 1.4180267e+00	 1.4558212e-01	[ 1.2142468e+00]	 2.0104289e-01
      90	 1.3698119e+00	 1.4701764e-01	 1.4214362e+00	 1.4528689e-01	[ 1.2177686e+00]	 1.9928575e-01


.. parsed-literal::

      91	 1.3746631e+00	 1.4631635e-01	 1.4264190e+00	 1.4511109e-01	[ 1.2183610e+00]	 2.0938230e-01


.. parsed-literal::

      92	 1.3790349e+00	 1.4622601e-01	 1.4309908e+00	 1.4482211e-01	[ 1.2247720e+00]	 2.0874119e-01


.. parsed-literal::

      93	 1.3839581e+00	 1.4524649e-01	 1.4358843e+00	 1.4422448e-01	[ 1.2303047e+00]	 2.1141243e-01


.. parsed-literal::

      94	 1.3877242e+00	 1.4425624e-01	 1.4396565e+00	 1.4375204e-01	[ 1.2341131e+00]	 2.1706486e-01


.. parsed-literal::

      95	 1.3916772e+00	 1.4347449e-01	 1.4436725e+00	 1.4342590e-01	[ 1.2382533e+00]	 2.1124005e-01


.. parsed-literal::

      96	 1.3953096e+00	 1.4217861e-01	 1.4476326e+00	 1.4266019e-01	  1.2344346e+00 	 2.1065879e-01


.. parsed-literal::

      97	 1.4007595e+00	 1.4231013e-01	 1.4529427e+00	 1.4275027e-01	[ 1.2435539e+00]	 2.1602869e-01
      98	 1.4034229e+00	 1.4234348e-01	 1.4556536e+00	 1.4255947e-01	[ 1.2461349e+00]	 1.8106890e-01


.. parsed-literal::

      99	 1.4071809e+00	 1.4217627e-01	 1.4595031e+00	 1.4189787e-01	[ 1.2489862e+00]	 2.1133065e-01


.. parsed-literal::

     100	 1.4113448e+00	 1.4158730e-01	 1.4638812e+00	 1.4050940e-01	[ 1.2572580e+00]	 2.0476890e-01


.. parsed-literal::

     101	 1.4164042e+00	 1.4081469e-01	 1.4689882e+00	 1.3968356e-01	[ 1.2582103e+00]	 2.1353936e-01


.. parsed-literal::

     102	 1.4198434e+00	 1.4005280e-01	 1.4724524e+00	 1.3910234e-01	[ 1.2606659e+00]	 2.0952082e-01


.. parsed-literal::

     103	 1.4231899e+00	 1.3890076e-01	 1.4759116e+00	 1.3906037e-01	[ 1.2607377e+00]	 2.0981288e-01
     104	 1.4264908e+00	 1.3846123e-01	 1.4793011e+00	 1.3891637e-01	[ 1.2646395e+00]	 1.8147421e-01


.. parsed-literal::

     105	 1.4296817e+00	 1.3827052e-01	 1.4825200e+00	 1.3908126e-01	[ 1.2673303e+00]	 2.1370983e-01


.. parsed-literal::

     106	 1.4347057e+00	 1.3789781e-01	 1.4876922e+00	 1.3928623e-01	[ 1.2708131e+00]	 2.2216058e-01
     107	 1.4377662e+00	 1.3745850e-01	 1.4909174e+00	 1.3913973e-01	[ 1.2711741e+00]	 1.9209552e-01


.. parsed-literal::

     108	 1.4415517e+00	 1.3730958e-01	 1.4945685e+00	 1.3892277e-01	[ 1.2742545e+00]	 2.0887923e-01


.. parsed-literal::

     109	 1.4449288e+00	 1.3662107e-01	 1.4979622e+00	 1.3826514e-01	[ 1.2758472e+00]	 2.1087003e-01
     110	 1.4471383e+00	 1.3615432e-01	 1.5001882e+00	 1.3793224e-01	[ 1.2768652e+00]	 1.8532467e-01


.. parsed-literal::

     111	 1.4510821e+00	 1.3487279e-01	 1.5042642e+00	 1.3730733e-01	[ 1.2786058e+00]	 1.7377567e-01
     112	 1.4549539e+00	 1.3429081e-01	 1.5081759e+00	 1.3698905e-01	[ 1.2810521e+00]	 1.9806814e-01


.. parsed-literal::

     113	 1.4572273e+00	 1.3437821e-01	 1.5104135e+00	 1.3704391e-01	[ 1.2838984e+00]	 2.1461177e-01


.. parsed-literal::

     114	 1.4604108e+00	 1.3426253e-01	 1.5137161e+00	 1.3707178e-01	[ 1.2839904e+00]	 2.0409775e-01


.. parsed-literal::

     115	 1.4629338e+00	 1.3424792e-01	 1.5163538e+00	 1.3674263e-01	[ 1.2842234e+00]	 2.1938252e-01
     116	 1.4658383e+00	 1.3408611e-01	 1.5192892e+00	 1.3612212e-01	[ 1.2852604e+00]	 2.0668101e-01


.. parsed-literal::

     117	 1.4696082e+00	 1.3352867e-01	 1.5232265e+00	 1.3527139e-01	  1.2798602e+00 	 1.9987464e-01
     118	 1.4714297e+00	 1.3341683e-01	 1.5250215e+00	 1.3460243e-01	[ 1.2858179e+00]	 2.0151496e-01


.. parsed-literal::

     119	 1.4728974e+00	 1.3328010e-01	 1.5264438e+00	 1.3465058e-01	[ 1.2876365e+00]	 2.0799375e-01


.. parsed-literal::

     120	 1.4770101e+00	 1.3256688e-01	 1.5306234e+00	 1.3473650e-01	[ 1.2894012e+00]	 2.1158600e-01


.. parsed-literal::

     121	 1.4794851e+00	 1.3225186e-01	 1.5331686e+00	 1.3470680e-01	[ 1.2908700e+00]	 2.0669937e-01


.. parsed-literal::

     122	 1.4837182e+00	 1.3183609e-01	 1.5375615e+00	 1.3478126e-01	  1.2887053e+00 	 2.1043181e-01


.. parsed-literal::

     123	 1.4874305e+00	 1.3146945e-01	 1.5414590e+00	 1.3459778e-01	  1.2847727e+00 	 2.1051455e-01


.. parsed-literal::

     124	 1.4906215e+00	 1.3155144e-01	 1.5445877e+00	 1.3429807e-01	  1.2861185e+00 	 2.1865773e-01


.. parsed-literal::

     125	 1.4926695e+00	 1.3166204e-01	 1.5465537e+00	 1.3426051e-01	  1.2885585e+00 	 2.1751881e-01
     126	 1.4954535e+00	 1.3174039e-01	 1.5493383e+00	 1.3435047e-01	[ 1.2915602e+00]	 1.9826841e-01


.. parsed-literal::

     127	 1.4981762e+00	 1.3159247e-01	 1.5521566e+00	 1.3399066e-01	[ 1.2943533e+00]	 1.8190169e-01


.. parsed-literal::

     128	 1.5008923e+00	 1.3142191e-01	 1.5548824e+00	 1.3389718e-01	[ 1.2973017e+00]	 2.1058869e-01
     129	 1.5039104e+00	 1.3111892e-01	 1.5579773e+00	 1.3360642e-01	[ 1.2976896e+00]	 1.9940925e-01


.. parsed-literal::

     130	 1.5059056e+00	 1.3095604e-01	 1.5600344e+00	 1.3318356e-01	[ 1.3001376e+00]	 1.8147135e-01


.. parsed-literal::

     131	 1.5084688e+00	 1.3072659e-01	 1.5626489e+00	 1.3275188e-01	  1.2931677e+00 	 2.0902634e-01


.. parsed-literal::

     132	 1.5108238e+00	 1.3049067e-01	 1.5650688e+00	 1.3213580e-01	  1.2869549e+00 	 2.1225429e-01
     133	 1.5125865e+00	 1.3025025e-01	 1.5668365e+00	 1.3167507e-01	  1.2831174e+00 	 1.8342137e-01


.. parsed-literal::

     134	 1.5145026e+00	 1.3001202e-01	 1.5687152e+00	 1.3140519e-01	  1.2833507e+00 	 2.0778537e-01
     135	 1.5169394e+00	 1.2959985e-01	 1.5711690e+00	 1.3087656e-01	  1.2831169e+00 	 1.9962263e-01


.. parsed-literal::

     136	 1.5187617e+00	 1.2927640e-01	 1.5730622e+00	 1.3052038e-01	  1.2828624e+00 	 1.9224191e-01


.. parsed-literal::

     137	 1.5212242e+00	 1.2897620e-01	 1.5756154e+00	 1.2994835e-01	  1.2790838e+00 	 2.0748854e-01


.. parsed-literal::

     138	 1.5238935e+00	 1.2875245e-01	 1.5783218e+00	 1.2956108e-01	  1.2749606e+00 	 2.1438241e-01


.. parsed-literal::

     139	 1.5258210e+00	 1.2844427e-01	 1.5803641e+00	 1.2935962e-01	  1.2671797e+00 	 2.2114730e-01


.. parsed-literal::

     140	 1.5281513e+00	 1.2838502e-01	 1.5826374e+00	 1.2934494e-01	  1.2666915e+00 	 2.1033430e-01


.. parsed-literal::

     141	 1.5298662e+00	 1.2835279e-01	 1.5843129e+00	 1.2943332e-01	  1.2686293e+00 	 2.2025108e-01


.. parsed-literal::

     142	 1.5314639e+00	 1.2818658e-01	 1.5859393e+00	 1.2955012e-01	  1.2658749e+00 	 2.1931577e-01


.. parsed-literal::

     143	 1.5332398e+00	 1.2807443e-01	 1.5877774e+00	 1.2957898e-01	  1.2653359e+00 	 2.1853209e-01


.. parsed-literal::

     144	 1.5348104e+00	 1.2791915e-01	 1.5893475e+00	 1.2960939e-01	  1.2617357e+00 	 2.1058464e-01
     145	 1.5365275e+00	 1.2771612e-01	 1.5911253e+00	 1.2962584e-01	  1.2569797e+00 	 1.7113900e-01


.. parsed-literal::

     146	 1.5377739e+00	 1.2759113e-01	 1.5924232e+00	 1.2961574e-01	  1.2507066e+00 	 1.8829513e-01


.. parsed-literal::

     147	 1.5396918e+00	 1.2743829e-01	 1.5943980e+00	 1.2967576e-01	  1.2455484e+00 	 2.3065138e-01


.. parsed-literal::

     148	 1.5418057e+00	 1.2710951e-01	 1.5965883e+00	 1.2954944e-01	  1.2370325e+00 	 2.1870446e-01
     149	 1.5437820e+00	 1.2687163e-01	 1.5986184e+00	 1.2961330e-01	  1.2303621e+00 	 1.7623258e-01


.. parsed-literal::

     150	 1.5453488e+00	 1.2670456e-01	 1.6001627e+00	 1.2951596e-01	  1.2292823e+00 	 2.1116304e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.29 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f238b291d80>



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
    CPU times: user 1.76 s, sys: 33 ms, total: 1.8 s
    Wall time: 557 ms


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

