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
       1	-3.3642732e-01	 3.1870289e-01	-3.2668505e-01	 3.2815806e-01	[-3.4429677e-01]	 4.6459150e-01


.. parsed-literal::

       2	-2.6878886e-01	 3.0902257e-01	-2.4589897e-01	 3.1775761e-01	[-2.7248474e-01]	 2.3094153e-01


.. parsed-literal::

       3	-2.2655593e-01	 2.8943786e-01	-1.8649069e-01	 2.9690140e-01	[-2.1773175e-01]	 2.7718043e-01
       4	-1.8770344e-01	 2.6614278e-01	-1.4531023e-01	 2.7070528e-01	[-1.6588110e-01]	 1.7915916e-01


.. parsed-literal::

       5	-1.0571315e-01	 2.5699035e-01	-7.2819878e-02	 2.6043980e-01	[-8.9148709e-02]	 2.0691013e-01


.. parsed-literal::

       6	-6.7833489e-02	 2.5102975e-01	-3.8343458e-02	 2.5446401e-01	[-5.0903516e-02]	 2.1387887e-01


.. parsed-literal::

       7	-5.0146759e-02	 2.4830953e-01	-2.6270375e-02	 2.5265418e-01	[-4.3508809e-02]	 2.0929050e-01


.. parsed-literal::

       8	-3.4211630e-02	 2.4548437e-01	-1.4437241e-02	 2.5099741e-01	[-3.7039636e-02]	 2.0522571e-01


.. parsed-literal::

       9	-2.1256732e-02	 2.4305344e-01	-4.1679467e-03	 2.4906339e-01	[-2.9808438e-02]	 2.0162630e-01
      10	-1.1969053e-02	 2.4131805e-01	 3.0592133e-03	 2.4622871e-01	[-1.6452201e-02]	 1.7703104e-01


.. parsed-literal::

      11	-5.2822653e-03	 2.4019954e-01	 8.7539626e-03	 2.4520407e-01	[-1.2971217e-02]	 2.0137429e-01


.. parsed-literal::

      12	-2.6235974e-03	 2.3975223e-01	 1.1247411e-02	 2.4489654e-01	[-1.0265082e-02]	 2.0711470e-01
      13	 1.0573553e-03	 2.3906264e-01	 1.4817734e-02	 2.4440745e-01	[-7.1702311e-03]	 1.8250823e-01


.. parsed-literal::

      14	 8.2092987e-03	 2.3742403e-01	 2.3348100e-02	 2.4280980e-01	[ 1.8425188e-03]	 1.9750476e-01


.. parsed-literal::

      15	 6.5477894e-02	 2.2564806e-01	 8.1993185e-02	 2.3001297e-01	[ 6.8884913e-02]	 3.1843352e-01


.. parsed-literal::

      16	 9.3786821e-02	 2.2373275e-01	 1.1251929e-01	 2.2999212e-01	[ 1.0037280e-01]	 2.0714521e-01
      17	 1.8903755e-01	 2.2002776e-01	 2.1068120e-01	 2.2435487e-01	[ 2.0185693e-01]	 1.9613171e-01


.. parsed-literal::

      18	 2.8670822e-01	 2.1636827e-01	 3.1709703e-01	 2.2037757e-01	[ 3.0713811e-01]	 1.9893217e-01
      19	 3.1603936e-01	 2.1779359e-01	 3.4571907e-01	 2.1878928e-01	[ 3.3848327e-01]	 1.9771314e-01


.. parsed-literal::

      20	 3.6945024e-01	 2.1296831e-01	 3.9996986e-01	 2.1412405e-01	[ 3.9111033e-01]	 2.1173692e-01
      21	 4.5062575e-01	 2.0881950e-01	 4.8166726e-01	 2.1031663e-01	[ 4.7299172e-01]	 1.9952035e-01


.. parsed-literal::

      22	 5.3960040e-01	 2.0808675e-01	 5.7335850e-01	 2.0991130e-01	[ 5.6721911e-01]	 1.9325542e-01
      23	 5.8099924e-01	 2.0585834e-01	 6.1940123e-01	 2.0929062e-01	[ 6.1539221e-01]	 1.8250990e-01


.. parsed-literal::

      24	 6.1633785e-01	 2.0299483e-01	 6.5331402e-01	 2.0558069e-01	[ 6.4725034e-01]	 2.0090985e-01
      25	 6.5681648e-01	 1.9990382e-01	 6.9276163e-01	 2.0334000e-01	[ 6.7956210e-01]	 1.8584776e-01


.. parsed-literal::

      26	 6.7688014e-01	 2.0486871e-01	 7.1131713e-01	 2.0987194e-01	[ 6.9067294e-01]	 2.0525146e-01


.. parsed-literal::

      27	 7.1161195e-01	 2.0664507e-01	 7.4582467e-01	 2.1257700e-01	[ 7.1964160e-01]	 2.0992613e-01
      28	 7.3626445e-01	 2.0077587e-01	 7.7242452e-01	 2.0761881e-01	[ 7.4873923e-01]	 1.8979955e-01


.. parsed-literal::

      29	 7.5906453e-01	 2.0180566e-01	 7.9615181e-01	 2.0912423e-01	[ 7.7063827e-01]	 2.0512342e-01


.. parsed-literal::

      30	 7.8611538e-01	 2.0253945e-01	 8.2397405e-01	 2.1039773e-01	[ 7.9781711e-01]	 2.1652293e-01
      31	 8.1480142e-01	 2.0235669e-01	 8.5323013e-01	 2.1115284e-01	[ 8.3116373e-01]	 1.9866824e-01


.. parsed-literal::

      32	 8.3998700e-01	 1.9784260e-01	 8.7892333e-01	 2.0424330e-01	[ 8.5607780e-01]	 2.1873212e-01


.. parsed-literal::

      33	 8.5697263e-01	 1.9443867e-01	 8.9655176e-01	 2.0002375e-01	[ 8.7574208e-01]	 2.1217442e-01


.. parsed-literal::

      34	 8.7622902e-01	 1.8964416e-01	 9.1668758e-01	 1.9421631e-01	[ 8.9849225e-01]	 2.1046805e-01


.. parsed-literal::

      35	 8.9474842e-01	 1.8838173e-01	 9.3608868e-01	 1.9195838e-01	[ 9.2167751e-01]	 2.1390557e-01


.. parsed-literal::

      36	 9.1359088e-01	 1.8562023e-01	 9.5540527e-01	 1.8974825e-01	[ 9.3867856e-01]	 2.1440005e-01
      37	 9.3029987e-01	 1.8431894e-01	 9.7267076e-01	 1.8933937e-01	[ 9.5137685e-01]	 1.9541550e-01


.. parsed-literal::

      38	 9.6110203e-01	 1.8500616e-01	 1.0042933e+00	 1.9187815e-01	[ 9.7619990e-01]	 1.8622398e-01


.. parsed-literal::

      39	 9.6418696e-01	 1.8944404e-01	 1.0092238e+00	 1.9679790e-01	  9.6059941e-01 	 2.0722699e-01


.. parsed-literal::

      40	 9.9123192e-01	 1.8513313e-01	 1.0352445e+00	 1.9286272e-01	[ 9.8780380e-01]	 2.0721912e-01


.. parsed-literal::

      41	 1.0011980e+00	 1.8313375e-01	 1.0454339e+00	 1.9125394e-01	[ 9.9872488e-01]	 2.1007299e-01
      42	 1.0093235e+00	 1.8199373e-01	 1.0538296e+00	 1.9036135e-01	[ 1.0030212e+00]	 1.7101312e-01


.. parsed-literal::

      43	 1.0250448e+00	 1.7986711e-01	 1.0707638e+00	 1.8817515e-01	[ 1.0090956e+00]	 2.1256828e-01
      44	 1.0355726e+00	 1.7810671e-01	 1.0813802e+00	 1.8698665e-01	[ 1.0209649e+00]	 2.0709968e-01


.. parsed-literal::

      45	 1.0459440e+00	 1.7676965e-01	 1.0915907e+00	 1.8518746e-01	[ 1.0343176e+00]	 2.1178293e-01


.. parsed-literal::

      46	 1.0608546e+00	 1.7477540e-01	 1.1068613e+00	 1.8261562e-01	[ 1.0478991e+00]	 2.1394706e-01


.. parsed-literal::

      47	 1.0701866e+00	 1.7417424e-01	 1.1161186e+00	 1.8221067e-01	[ 1.0542367e+00]	 2.0164537e-01


.. parsed-literal::

      48	 1.0801212e+00	 1.7344042e-01	 1.1260476e+00	 1.8154938e-01	[ 1.0564508e+00]	 2.2724724e-01


.. parsed-literal::

      49	 1.0890054e+00	 1.7293598e-01	 1.1351543e+00	 1.8162829e-01	[ 1.0596396e+00]	 2.0258284e-01


.. parsed-literal::

      50	 1.0981238e+00	 1.7153346e-01	 1.1444432e+00	 1.8007403e-01	[ 1.0726103e+00]	 2.0756507e-01


.. parsed-literal::

      51	 1.1075023e+00	 1.7061969e-01	 1.1538658e+00	 1.7915151e-01	[ 1.0784155e+00]	 2.1010780e-01


.. parsed-literal::

      52	 1.1159703e+00	 1.6865302e-01	 1.1625110e+00	 1.7676101e-01	[ 1.0877500e+00]	 2.0982337e-01


.. parsed-literal::

      53	 1.1262641e+00	 1.6474895e-01	 1.1732943e+00	 1.7197037e-01	[ 1.0938424e+00]	 2.0885706e-01
      54	 1.1329225e+00	 1.6110935e-01	 1.1805845e+00	 1.6728370e-01	[ 1.1045955e+00]	 1.8093657e-01


.. parsed-literal::

      55	 1.1419291e+00	 1.5961574e-01	 1.1894853e+00	 1.6590475e-01	[ 1.1061860e+00]	 2.0223641e-01


.. parsed-literal::

      56	 1.1492500e+00	 1.5818000e-01	 1.1969270e+00	 1.6489188e-01	  1.1011951e+00 	 2.1663642e-01
      57	 1.1557127e+00	 1.5717829e-01	 1.2035885e+00	 1.6422312e-01	  1.0983863e+00 	 1.9091439e-01


.. parsed-literal::

      58	 1.1629354e+00	 1.5569367e-01	 1.2114126e+00	 1.6371837e-01	  1.0838288e+00 	 2.0922709e-01
      59	 1.1717274e+00	 1.5411989e-01	 1.2203090e+00	 1.6224834e-01	  1.0879404e+00 	 1.8062949e-01


.. parsed-literal::

      60	 1.1765358e+00	 1.5400854e-01	 1.2250039e+00	 1.6214455e-01	  1.0932129e+00 	 2.1498752e-01


.. parsed-literal::

      61	 1.1823559e+00	 1.5251319e-01	 1.2311033e+00	 1.6055781e-01	  1.0902344e+00 	 2.0566058e-01


.. parsed-literal::

      62	 1.1878048e+00	 1.5152233e-01	 1.2367327e+00	 1.5970126e-01	  1.1009163e+00 	 2.0721769e-01


.. parsed-literal::

      63	 1.1937625e+00	 1.5025829e-01	 1.2428319e+00	 1.5834948e-01	[ 1.1107985e+00]	 2.1058416e-01


.. parsed-literal::

      64	 1.1996373e+00	 1.4887614e-01	 1.2489525e+00	 1.5657270e-01	[ 1.1230169e+00]	 2.1582270e-01


.. parsed-literal::

      65	 1.2059061e+00	 1.4775524e-01	 1.2553917e+00	 1.5519053e-01	  1.1218716e+00 	 2.1397138e-01
      66	 1.2109484e+00	 1.4728046e-01	 1.2604419e+00	 1.5467803e-01	  1.1212315e+00 	 1.8369722e-01


.. parsed-literal::

      67	 1.2184259e+00	 1.4641500e-01	 1.2681232e+00	 1.5376089e-01	  1.1226078e+00 	 1.9766331e-01


.. parsed-literal::

      68	 1.2251107e+00	 1.4563758e-01	 1.2752747e+00	 1.5327565e-01	  1.1118921e+00 	 2.1304750e-01
      69	 1.2313593e+00	 1.4489202e-01	 1.2815953e+00	 1.5254804e-01	  1.1154404e+00 	 1.9751859e-01


.. parsed-literal::

      70	 1.2371366e+00	 1.4372200e-01	 1.2877493e+00	 1.5138220e-01	  1.1148303e+00 	 2.0452976e-01


.. parsed-literal::

      71	 1.2428846e+00	 1.4315019e-01	 1.2936936e+00	 1.5095174e-01	  1.1144607e+00 	 2.0757937e-01


.. parsed-literal::

      72	 1.2488810e+00	 1.4281049e-01	 1.2999273e+00	 1.5078371e-01	  1.1107315e+00 	 2.1159387e-01


.. parsed-literal::

      73	 1.2529830e+00	 1.4234148e-01	 1.3041509e+00	 1.5051288e-01	  1.1046768e+00 	 2.0627284e-01


.. parsed-literal::

      74	 1.2570654e+00	 1.4247028e-01	 1.3081240e+00	 1.5075378e-01	  1.1022147e+00 	 2.0117617e-01


.. parsed-literal::

      75	 1.2626247e+00	 1.4230715e-01	 1.3136223e+00	 1.5074727e-01	  1.1054667e+00 	 2.1510410e-01


.. parsed-literal::

      76	 1.2689674e+00	 1.4182453e-01	 1.3202505e+00	 1.5058670e-01	  1.0909677e+00 	 2.3650050e-01


.. parsed-literal::

      77	 1.2749401e+00	 1.4138595e-01	 1.3261219e+00	 1.5044255e-01	  1.1017667e+00 	 2.1378636e-01
      78	 1.2794836e+00	 1.4078990e-01	 1.3307574e+00	 1.5005822e-01	  1.1070457e+00 	 1.9847059e-01


.. parsed-literal::

      79	 1.2857499e+00	 1.4006250e-01	 1.3370487e+00	 1.4962355e-01	  1.1152416e+00 	 2.0831132e-01


.. parsed-literal::

      80	 1.2911933e+00	 1.3895039e-01	 1.3428766e+00	 1.4906960e-01	  1.1175205e+00 	 2.1252394e-01


.. parsed-literal::

      81	 1.2984928e+00	 1.3865347e-01	 1.3500097e+00	 1.4887285e-01	  1.1182910e+00 	 2.2012424e-01


.. parsed-literal::

      82	 1.3025238e+00	 1.3870828e-01	 1.3538789e+00	 1.4896179e-01	  1.1199028e+00 	 2.0633101e-01
      83	 1.3089673e+00	 1.3829751e-01	 1.3604036e+00	 1.4876001e-01	  1.1194281e+00 	 1.7116785e-01


.. parsed-literal::

      84	 1.3107993e+00	 1.3811373e-01	 1.3627469e+00	 1.4881572e-01	  1.1023627e+00 	 2.1611142e-01
      85	 1.3179382e+00	 1.3777529e-01	 1.3697004e+00	 1.4839507e-01	  1.1215529e+00 	 1.8407226e-01


.. parsed-literal::

      86	 1.3215183e+00	 1.3733181e-01	 1.3734039e+00	 1.4797177e-01	[ 1.1291038e+00]	 2.1220565e-01


.. parsed-literal::

      87	 1.3253585e+00	 1.3690841e-01	 1.3774810e+00	 1.4750365e-01	[ 1.1325463e+00]	 2.1022534e-01


.. parsed-literal::

      88	 1.3312078e+00	 1.3624971e-01	 1.3836288e+00	 1.4677357e-01	  1.1305098e+00 	 2.0864463e-01


.. parsed-literal::

      89	 1.3346201e+00	 1.3614280e-01	 1.3872770e+00	 1.4641316e-01	  1.1308364e+00 	 2.9061079e-01
      90	 1.3388219e+00	 1.3579547e-01	 1.3914811e+00	 1.4606166e-01	  1.1253725e+00 	 2.0022869e-01


.. parsed-literal::

      91	 1.3423785e+00	 1.3555969e-01	 1.3949780e+00	 1.4580855e-01	  1.1219852e+00 	 1.9348574e-01


.. parsed-literal::

      92	 1.3468417e+00	 1.3516766e-01	 1.3995130e+00	 1.4540387e-01	  1.1194030e+00 	 2.1120715e-01
      93	 1.3512530e+00	 1.3500431e-01	 1.4039984e+00	 1.4504358e-01	  1.1230690e+00 	 1.7237234e-01


.. parsed-literal::

      94	 1.3547785e+00	 1.3490416e-01	 1.4075498e+00	 1.4473091e-01	  1.1270563e+00 	 2.0267725e-01


.. parsed-literal::

      95	 1.3594550e+00	 1.3472552e-01	 1.4123666e+00	 1.4422290e-01	  1.1325224e+00 	 2.0814085e-01
      96	 1.3635631e+00	 1.3461298e-01	 1.4164618e+00	 1.4385570e-01	  1.1291433e+00 	 2.0344400e-01


.. parsed-literal::

      97	 1.3676945e+00	 1.3436685e-01	 1.4206108e+00	 1.4350172e-01	  1.1221010e+00 	 1.9890809e-01


.. parsed-literal::

      98	 1.3740004e+00	 1.3377813e-01	 1.4270355e+00	 1.4281586e-01	  1.1120495e+00 	 2.1714234e-01


.. parsed-literal::

      99	 1.3767887e+00	 1.3351127e-01	 1.4298465e+00	 1.4259661e-01	  1.1042780e+00 	 2.9974794e-01


.. parsed-literal::

     100	 1.3799080e+00	 1.3321633e-01	 1.4329701e+00	 1.4231994e-01	  1.1054071e+00 	 2.0504355e-01
     101	 1.3833863e+00	 1.3275937e-01	 1.4364732e+00	 1.4194714e-01	  1.1093767e+00 	 1.9835138e-01


.. parsed-literal::

     102	 1.3852942e+00	 1.3279092e-01	 1.4384533e+00	 1.4187115e-01	  1.1114652e+00 	 2.0186234e-01


.. parsed-literal::

     103	 1.3880894e+00	 1.3249640e-01	 1.4412429e+00	 1.4167211e-01	  1.1169572e+00 	 2.2152925e-01


.. parsed-literal::

     104	 1.3906961e+00	 1.3223686e-01	 1.4438796e+00	 1.4153556e-01	  1.1197425e+00 	 2.1713090e-01


.. parsed-literal::

     105	 1.3935747e+00	 1.3190410e-01	 1.4468281e+00	 1.4143993e-01	  1.1179457e+00 	 2.0833755e-01


.. parsed-literal::

     106	 1.3988517e+00	 1.3128779e-01	 1.4521682e+00	 1.4141231e-01	  1.1097848e+00 	 2.1387577e-01


.. parsed-literal::

     107	 1.4015447e+00	 1.3072389e-01	 1.4550095e+00	 1.4149741e-01	  1.1060020e+00 	 3.2225370e-01


.. parsed-literal::

     108	 1.4056601e+00	 1.3030092e-01	 1.4591467e+00	 1.4166393e-01	  1.0994495e+00 	 2.0689106e-01
     109	 1.4083071e+00	 1.3015216e-01	 1.4617909e+00	 1.4167556e-01	  1.1028880e+00 	 2.0319223e-01


.. parsed-literal::

     110	 1.4120566e+00	 1.2980713e-01	 1.4656809e+00	 1.4186457e-01	  1.1045315e+00 	 1.9956541e-01


.. parsed-literal::

     111	 1.4155642e+00	 1.2964988e-01	 1.4693403e+00	 1.4175117e-01	  1.1138133e+00 	 2.1376705e-01
     112	 1.4184374e+00	 1.2950269e-01	 1.4722641e+00	 1.4158334e-01	  1.1188313e+00 	 1.9987416e-01


.. parsed-literal::

     113	 1.4220743e+00	 1.2934925e-01	 1.4760382e+00	 1.4163092e-01	  1.1196873e+00 	 2.0825124e-01


.. parsed-literal::

     114	 1.4244047e+00	 1.2923266e-01	 1.4783640e+00	 1.4124203e-01	  1.1237069e+00 	 2.1229529e-01
     115	 1.4274928e+00	 1.2902274e-01	 1.4814405e+00	 1.4101832e-01	  1.1244431e+00 	 1.7676711e-01


.. parsed-literal::

     116	 1.4296904e+00	 1.2879120e-01	 1.4838480e+00	 1.4092814e-01	  1.1080163e+00 	 1.8374491e-01


.. parsed-literal::

     117	 1.4329366e+00	 1.2846772e-01	 1.4869963e+00	 1.4063947e-01	  1.1142088e+00 	 2.1016097e-01


.. parsed-literal::

     118	 1.4339826e+00	 1.2843154e-01	 1.4880539e+00	 1.4068197e-01	  1.1153637e+00 	 2.1285033e-01


.. parsed-literal::

     119	 1.4373797e+00	 1.2820734e-01	 1.4916044e+00	 1.4072090e-01	  1.1148741e+00 	 2.0544934e-01


.. parsed-literal::

     120	 1.4395416e+00	 1.2785521e-01	 1.4940084e+00	 1.4042478e-01	  1.1174950e+00 	 2.0839143e-01


.. parsed-literal::

     121	 1.4426694e+00	 1.2774808e-01	 1.4970282e+00	 1.4037017e-01	  1.1215566e+00 	 2.0904231e-01
     122	 1.4445892e+00	 1.2764831e-01	 1.4988588e+00	 1.4022508e-01	  1.1287044e+00 	 1.9843221e-01


.. parsed-literal::

     123	 1.4463485e+00	 1.2753826e-01	 1.5005396e+00	 1.3999713e-01	[ 1.1364012e+00]	 1.8913007e-01


.. parsed-literal::

     124	 1.4497702e+00	 1.2735824e-01	 1.5038676e+00	 1.3993351e-01	[ 1.1429774e+00]	 2.1123743e-01


.. parsed-literal::

     125	 1.4527750e+00	 1.2720367e-01	 1.5068344e+00	 1.3918956e-01	[ 1.1486047e+00]	 2.1142673e-01


.. parsed-literal::

     126	 1.4553428e+00	 1.2710854e-01	 1.5093847e+00	 1.3925620e-01	  1.1438902e+00 	 2.1770191e-01
     127	 1.4577899e+00	 1.2706106e-01	 1.5119351e+00	 1.3929766e-01	  1.1324835e+00 	 2.0735383e-01


.. parsed-literal::

     128	 1.4600414e+00	 1.2705455e-01	 1.5143016e+00	 1.3906777e-01	  1.1265637e+00 	 2.0883656e-01
     129	 1.4625159e+00	 1.2718808e-01	 1.5169547e+00	 1.3883616e-01	  1.1179329e+00 	 1.7361164e-01


.. parsed-literal::

     130	 1.4643362e+00	 1.2721830e-01	 1.5186942e+00	 1.3868393e-01	  1.1226888e+00 	 2.0726013e-01
     131	 1.4657894e+00	 1.2713938e-01	 1.5200765e+00	 1.3839925e-01	  1.1324240e+00 	 1.9865227e-01


.. parsed-literal::

     132	 1.4680509e+00	 1.2694025e-01	 1.5223167e+00	 1.3797577e-01	  1.1372780e+00 	 2.0391202e-01


.. parsed-literal::

     133	 1.4701748e+00	 1.2669654e-01	 1.5245877e+00	 1.3769747e-01	  1.1450182e+00 	 2.1558738e-01
     134	 1.4725018e+00	 1.2653030e-01	 1.5269017e+00	 1.3765720e-01	  1.1388186e+00 	 2.0296693e-01


.. parsed-literal::

     135	 1.4742753e+00	 1.2643036e-01	 1.5287620e+00	 1.3777412e-01	  1.1300757e+00 	 2.0384169e-01


.. parsed-literal::

     136	 1.4765645e+00	 1.2630208e-01	 1.5312031e+00	 1.3789716e-01	  1.1218311e+00 	 2.1243095e-01
     137	 1.4790460e+00	 1.2614930e-01	 1.5339910e+00	 1.3806982e-01	  1.1066162e+00 	 2.0338488e-01


.. parsed-literal::

     138	 1.4818955e+00	 1.2597196e-01	 1.5368619e+00	 1.3819889e-01	  1.0971429e+00 	 2.1762633e-01


.. parsed-literal::

     139	 1.4834786e+00	 1.2587979e-01	 1.5383377e+00	 1.3821337e-01	  1.0999609e+00 	 2.1185756e-01
     140	 1.4857317e+00	 1.2566440e-01	 1.5405417e+00	 1.3831490e-01	  1.0919125e+00 	 1.9809675e-01


.. parsed-literal::

     141	 1.4870302e+00	 1.2534377e-01	 1.5420296e+00	 1.3857702e-01	  1.0807262e+00 	 1.9311714e-01


.. parsed-literal::

     142	 1.4894157e+00	 1.2527792e-01	 1.5443382e+00	 1.3858830e-01	  1.0745782e+00 	 2.0637822e-01
     143	 1.4907003e+00	 1.2516725e-01	 1.5456937e+00	 1.3860689e-01	  1.0667809e+00 	 1.7326880e-01


.. parsed-literal::

     144	 1.4920051e+00	 1.2502195e-01	 1.5470764e+00	 1.3855436e-01	  1.0597930e+00 	 2.1613646e-01


.. parsed-literal::

     145	 1.4945504e+00	 1.2478342e-01	 1.5497386e+00	 1.3848288e-01	  1.0397936e+00 	 2.1572328e-01


.. parsed-literal::

     146	 1.4961035e+00	 1.2458077e-01	 1.5513426e+00	 1.3822889e-01	  1.0307575e+00 	 3.0245113e-01
     147	 1.4979703e+00	 1.2445983e-01	 1.5531905e+00	 1.3811558e-01	  1.0151069e+00 	 2.1068311e-01


.. parsed-literal::

     148	 1.4996631e+00	 1.2442015e-01	 1.5548515e+00	 1.3815749e-01	  1.0019679e+00 	 2.1662116e-01
     149	 1.5012482e+00	 1.2430011e-01	 1.5564320e+00	 1.3789645e-01	  9.7668679e-01 	 2.0267582e-01


.. parsed-literal::

     150	 1.5028442e+00	 1.2427507e-01	 1.5580321e+00	 1.3796460e-01	  9.7296661e-01 	 1.8936610e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 988 ms, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f7bfc947730>



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
    CPU times: user 2 s, sys: 47.9 ms, total: 2.05 s
    Wall time: 611 ms


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

