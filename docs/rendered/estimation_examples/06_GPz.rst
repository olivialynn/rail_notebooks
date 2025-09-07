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
       1	-3.3841960e-01	 3.1884157e-01	-3.2873201e-01	 3.2681284e-01	[-3.4438016e-01]	 4.6480131e-01


.. parsed-literal::

       2	-2.6645800e-01	 3.0781616e-01	-2.4186105e-01	 3.1579744e-01	[-2.6687437e-01]	 2.3256922e-01


.. parsed-literal::

       3	-2.2206590e-01	 2.8717116e-01	-1.7906021e-01	 2.9550828e-01	[-2.1614399e-01]	 2.7814960e-01
       4	-1.9074269e-01	 2.6360666e-01	-1.4970036e-01	 2.7357888e-01	[-2.0969476e-01]	 1.7153382e-01


.. parsed-literal::

       5	-9.8238751e-02	 2.5554139e-01	-6.2658994e-02	 2.6269504e-01	[-9.7433588e-02]	 2.1282673e-01


.. parsed-literal::

       6	-6.6805122e-02	 2.5075323e-01	-3.6182911e-02	 2.5672621e-01	[-5.9701493e-02]	 2.1751285e-01


.. parsed-literal::

       7	-4.8176758e-02	 2.4782362e-01	-2.4324976e-02	 2.5382599e-01	[-4.8447929e-02]	 2.1045184e-01
       8	-3.7352753e-02	 2.4607123e-01	-1.7024199e-02	 2.5157116e-01	[-3.9276440e-02]	 1.8863988e-01


.. parsed-literal::

       9	-2.4053804e-02	 2.4362314e-01	-6.6112719e-03	 2.4857143e-01	[-2.6198841e-02]	 2.0976949e-01


.. parsed-literal::

      10	-1.1584595e-02	 2.4110909e-01	 3.9132045e-03	 2.4678077e-01	[-1.8997499e-02]	 2.0998693e-01


.. parsed-literal::

      11	-9.3506208e-03	 2.4063206e-01	 4.9483916e-03	 2.4691393e-01	[-1.7562326e-02]	 2.1132302e-01
      12	-3.9723585e-03	 2.3993883e-01	 1.0115824e-02	 2.4613114e-01	[-1.4468933e-02]	 1.8804049e-01


.. parsed-literal::

      13	-1.5023231e-03	 2.3941466e-01	 1.2579218e-02	 2.4556700e-01	[-1.2448322e-02]	 2.1018696e-01
      14	 2.1612676e-03	 2.3862157e-01	 1.6506373e-02	 2.4470357e-01	[-8.9582576e-03]	 1.9697595e-01


.. parsed-literal::

      15	 8.9764258e-02	 2.2558679e-01	 1.1040224e-01	 2.2934815e-01	[ 9.0031254e-02]	 3.0740285e-01


.. parsed-literal::

      16	 1.1756294e-01	 2.2215289e-01	 1.3801467e-01	 2.2694382e-01	[ 1.1831714e-01]	 4.4201374e-01


.. parsed-literal::

      17	 1.6384928e-01	 2.1631739e-01	 1.8565201e-01	 2.2277500e-01	[ 1.6352410e-01]	 2.1825981e-01


.. parsed-literal::

      18	 2.5917959e-01	 2.1593055e-01	 2.8637225e-01	 2.2141182e-01	[ 2.6026186e-01]	 2.1793699e-01
      19	 3.4324311e-01	 2.1124080e-01	 3.7423326e-01	 2.1843585e-01	[ 3.3110620e-01]	 1.9938755e-01


.. parsed-literal::

      20	 3.7908298e-01	 2.1036670e-01	 4.1192653e-01	 2.2046335e-01	[ 3.4841225e-01]	 2.0175648e-01


.. parsed-literal::

      21	 4.3416906e-01	 2.0398188e-01	 4.6765953e-01	 2.1442749e-01	[ 4.0414279e-01]	 2.1829510e-01


.. parsed-literal::

      22	 5.0081895e-01	 1.9941122e-01	 5.3439759e-01	 2.0665060e-01	[ 4.7984259e-01]	 2.1392655e-01
      23	 6.1301534e-01	 1.9505860e-01	 6.4896141e-01	 2.0293300e-01	[ 6.0654214e-01]	 1.8866897e-01


.. parsed-literal::

      24	 6.6974203e-01	 1.9161018e-01	 7.0854381e-01	 1.9936727e-01	[ 6.8616161e-01]	 2.1806741e-01


.. parsed-literal::

      25	 7.1892326e-01	 1.9023100e-01	 7.5736786e-01	 1.9849957e-01	[ 7.3023773e-01]	 2.0421505e-01
      26	 7.4980727e-01	 1.8967599e-01	 7.8685642e-01	 1.9795722e-01	[ 7.4769760e-01]	 1.7743945e-01


.. parsed-literal::

      27	 7.7651109e-01	 1.9026434e-01	 8.1387289e-01	 1.9904793e-01	[ 7.6330890e-01]	 2.0856977e-01


.. parsed-literal::

      28	 8.0032101e-01	 1.8953050e-01	 8.3755600e-01	 1.9951029e-01	[ 7.8202846e-01]	 3.1980038e-01


.. parsed-literal::

      29	 8.3567560e-01	 1.8561553e-01	 8.7406487e-01	 1.9586064e-01	[ 8.1545487e-01]	 2.0896816e-01
      30	 8.7326106e-01	 1.8263432e-01	 9.1411142e-01	 1.9279611e-01	[ 8.5310233e-01]	 2.0761919e-01


.. parsed-literal::

      31	 8.9008826e-01	 1.8562362e-01	 9.3220561e-01	 1.9258215e-01	[ 8.6666419e-01]	 2.0808125e-01


.. parsed-literal::

      32	 9.0869897e-01	 1.8320696e-01	 9.5050770e-01	 1.9115012e-01	[ 8.8493217e-01]	 2.0990467e-01
      33	 9.2204690e-01	 1.8119316e-01	 9.6424132e-01	 1.8973393e-01	[ 8.9851466e-01]	 1.6468859e-01


.. parsed-literal::

      34	 9.4325466e-01	 1.7994519e-01	 9.8600026e-01	 1.8744297e-01	[ 9.1794089e-01]	 2.1746182e-01
      35	 9.5549058e-01	 1.8072792e-01	 9.9977673e-01	 1.8503053e-01	[ 9.4078648e-01]	 1.7846465e-01


.. parsed-literal::

      36	 9.7683632e-01	 1.7879955e-01	 1.0203300e+00	 1.8446612e-01	[ 9.5874758e-01]	 1.9885230e-01


.. parsed-literal::

      37	 9.8417134e-01	 1.7659058e-01	 1.0274293e+00	 1.8250164e-01	[ 9.6954081e-01]	 2.1020889e-01
      38	 9.9346161e-01	 1.7594551e-01	 1.0368914e+00	 1.8195913e-01	[ 9.7980819e-01]	 1.7669487e-01


.. parsed-literal::

      39	 1.0100429e+00	 1.7410897e-01	 1.0545636e+00	 1.8046365e-01	[ 1.0015257e+00]	 1.9776273e-01
      40	 1.0189510e+00	 1.7460744e-01	 1.0647560e+00	 1.8094303e-01	[ 1.0127678e+00]	 1.9785237e-01


.. parsed-literal::

      41	 1.0292122e+00	 1.7358581e-01	 1.0748164e+00	 1.7998961e-01	[ 1.0217523e+00]	 1.8335223e-01
      42	 1.0383710e+00	 1.7266687e-01	 1.0842009e+00	 1.7914734e-01	[ 1.0289306e+00]	 2.0432186e-01


.. parsed-literal::

      43	 1.0481422e+00	 1.7169857e-01	 1.0942033e+00	 1.7817161e-01	[ 1.0357117e+00]	 1.7292380e-01


.. parsed-literal::

      44	 1.0634945e+00	 1.7081755e-01	 1.1106218e+00	 1.7694896e-01	[ 1.0437293e+00]	 2.0303845e-01


.. parsed-literal::

      45	 1.0738865e+00	 1.6894008e-01	 1.1208647e+00	 1.7490531e-01	[ 1.0528743e+00]	 2.0504236e-01


.. parsed-literal::

      46	 1.0849873e+00	 1.6779718e-01	 1.1318600e+00	 1.7321859e-01	[ 1.0668451e+00]	 2.1059227e-01


.. parsed-literal::

      47	 1.0937525e+00	 1.6694513e-01	 1.1409240e+00	 1.7157666e-01	[ 1.0752229e+00]	 2.1239781e-01
      48	 1.1003924e+00	 1.6630424e-01	 1.1476923e+00	 1.7089532e-01	[ 1.0821218e+00]	 2.0560670e-01


.. parsed-literal::

      49	 1.1105113e+00	 1.6554558e-01	 1.1578848e+00	 1.7019237e-01	[ 1.0892745e+00]	 2.1790600e-01


.. parsed-literal::

      50	 1.1206242e+00	 1.6472087e-01	 1.1682208e+00	 1.6948899e-01	[ 1.0978509e+00]	 2.1515274e-01


.. parsed-literal::

      51	 1.1300494e+00	 1.6452428e-01	 1.1781443e+00	 1.6951856e-01	[ 1.1046036e+00]	 2.1053433e-01
      52	 1.1376534e+00	 1.6389921e-01	 1.1858164e+00	 1.6836854e-01	[ 1.1145672e+00]	 1.8155646e-01


.. parsed-literal::

      53	 1.1445738e+00	 1.6362935e-01	 1.1929475e+00	 1.6752936e-01	[ 1.1241623e+00]	 2.0413876e-01


.. parsed-literal::

      54	 1.1518702e+00	 1.6357014e-01	 1.2005114e+00	 1.6681496e-01	[ 1.1309426e+00]	 2.1468472e-01


.. parsed-literal::

      55	 1.1602768e+00	 1.6372362e-01	 1.2090715e+00	 1.6651092e-01	[ 1.1367090e+00]	 2.0851421e-01


.. parsed-literal::

      56	 1.1706531e+00	 1.6322450e-01	 1.2201330e+00	 1.6616080e-01	[ 1.1432617e+00]	 2.1239305e-01
      57	 1.1771841e+00	 1.6240521e-01	 1.2267091e+00	 1.6475977e-01	[ 1.1471582e+00]	 1.8629670e-01


.. parsed-literal::

      58	 1.1824265e+00	 1.6104302e-01	 1.2316717e+00	 1.6353661e-01	[ 1.1540067e+00]	 1.9813776e-01


.. parsed-literal::

      59	 1.1881624e+00	 1.5979668e-01	 1.2375571e+00	 1.6233045e-01	[ 1.1616198e+00]	 2.1526122e-01
      60	 1.1950092e+00	 1.5863392e-01	 1.2445608e+00	 1.6070121e-01	[ 1.1681345e+00]	 1.9226265e-01


.. parsed-literal::

      61	 1.2037224e+00	 1.5756417e-01	 1.2534173e+00	 1.5927383e-01	[ 1.1770060e+00]	 2.0208001e-01


.. parsed-literal::

      62	 1.2114636e+00	 1.5711559e-01	 1.2613144e+00	 1.5841021e-01	[ 1.1799727e+00]	 2.0731378e-01


.. parsed-literal::

      63	 1.2171575e+00	 1.5681949e-01	 1.2669989e+00	 1.5810236e-01	[ 1.1845380e+00]	 2.1811533e-01


.. parsed-literal::

      64	 1.2281666e+00	 1.5574467e-01	 1.2785614e+00	 1.5723477e-01	[ 1.1932826e+00]	 2.0800829e-01


.. parsed-literal::

      65	 1.2335727e+00	 1.5474599e-01	 1.2841350e+00	 1.5621748e-01	[ 1.2009119e+00]	 2.1433830e-01


.. parsed-literal::

      66	 1.2410751e+00	 1.5407710e-01	 1.2915293e+00	 1.5571126e-01	[ 1.2090681e+00]	 2.1428251e-01


.. parsed-literal::

      67	 1.2472563e+00	 1.5344873e-01	 1.2978847e+00	 1.5528488e-01	[ 1.2162494e+00]	 2.1369505e-01
      68	 1.2529933e+00	 1.5311895e-01	 1.3037594e+00	 1.5509587e-01	[ 1.2223684e+00]	 1.8118453e-01


.. parsed-literal::

      69	 1.2640594e+00	 1.5209020e-01	 1.3153798e+00	 1.5402309e-01	[ 1.2291732e+00]	 1.8529558e-01


.. parsed-literal::

      70	 1.2689532e+00	 1.5414197e-01	 1.3206468e+00	 1.5616182e-01	[ 1.2301062e+00]	 2.1175265e-01
      71	 1.2774939e+00	 1.5269078e-01	 1.3287419e+00	 1.5426619e-01	[ 1.2422491e+00]	 1.7099071e-01


.. parsed-literal::

      72	 1.2816130e+00	 1.5197313e-01	 1.3328897e+00	 1.5334223e-01	[ 1.2457178e+00]	 1.8140626e-01
      73	 1.2875418e+00	 1.5137646e-01	 1.3388962e+00	 1.5249041e-01	[ 1.2510303e+00]	 2.0191908e-01


.. parsed-literal::

      74	 1.2970537e+00	 1.4963765e-01	 1.3485748e+00	 1.5009403e-01	[ 1.2584874e+00]	 1.9934821e-01
      75	 1.3052708e+00	 1.4913369e-01	 1.3570960e+00	 1.4913035e-01	[ 1.2674915e+00]	 2.0043182e-01


.. parsed-literal::

      76	 1.3124697e+00	 1.4847011e-01	 1.3643259e+00	 1.4842898e-01	[ 1.2689769e+00]	 2.0012593e-01
      77	 1.3209198e+00	 1.4760985e-01	 1.3730703e+00	 1.4741799e-01	[ 1.2694761e+00]	 1.9845700e-01


.. parsed-literal::

      78	 1.3269655e+00	 1.4716539e-01	 1.3792765e+00	 1.4753984e-01	  1.2687586e+00 	 2.0684600e-01
      79	 1.3336626e+00	 1.4690144e-01	 1.3861713e+00	 1.4754246e-01	[ 1.2695969e+00]	 1.9001746e-01


.. parsed-literal::

      80	 1.3399691e+00	 1.4672417e-01	 1.3927140e+00	 1.4776694e-01	  1.2692126e+00 	 2.0835042e-01
      81	 1.3449663e+00	 1.4674378e-01	 1.3978377e+00	 1.4827104e-01	  1.2694687e+00 	 1.8030429e-01


.. parsed-literal::

      82	 1.3497977e+00	 1.4647923e-01	 1.4026712e+00	 1.4793321e-01	[ 1.2737924e+00]	 1.9895291e-01


.. parsed-literal::

      83	 1.3552441e+00	 1.4607441e-01	 1.4082596e+00	 1.4762097e-01	  1.2735128e+00 	 2.1131539e-01
      84	 1.3598753e+00	 1.4506844e-01	 1.4130384e+00	 1.4675947e-01	  1.2735919e+00 	 1.9150925e-01


.. parsed-literal::

      85	 1.3652819e+00	 1.4438568e-01	 1.4185729e+00	 1.4658229e-01	[ 1.2777081e+00]	 1.6772318e-01
      86	 1.3703698e+00	 1.4372219e-01	 1.4237648e+00	 1.4647142e-01	[ 1.2810948e+00]	 1.9768000e-01


.. parsed-literal::

      87	 1.3759528e+00	 1.4269278e-01	 1.4296731e+00	 1.4573329e-01	[ 1.2902264e+00]	 2.0833492e-01


.. parsed-literal::

      88	 1.3808312e+00	 1.4208287e-01	 1.4345747e+00	 1.4575014e-01	[ 1.2961017e+00]	 2.1167064e-01
      89	 1.3848441e+00	 1.4170369e-01	 1.4384919e+00	 1.4534010e-01	[ 1.2987611e+00]	 1.8769741e-01


.. parsed-literal::

      90	 1.3919359e+00	 1.4073565e-01	 1.4455545e+00	 1.4430502e-01	[ 1.3076051e+00]	 2.1181607e-01
      91	 1.3952924e+00	 1.4009584e-01	 1.4489675e+00	 1.4422070e-01	  1.3031137e+00 	 1.9111657e-01


.. parsed-literal::

      92	 1.3991462e+00	 1.3992990e-01	 1.4526965e+00	 1.4409379e-01	[ 1.3086581e+00]	 2.0843410e-01


.. parsed-literal::

      93	 1.4031166e+00	 1.3955136e-01	 1.4566564e+00	 1.4410359e-01	[ 1.3106765e+00]	 2.1293163e-01
      94	 1.4066667e+00	 1.3930382e-01	 1.4601442e+00	 1.4395370e-01	  1.3102268e+00 	 1.8637562e-01


.. parsed-literal::

      95	 1.4127739e+00	 1.3853331e-01	 1.4663335e+00	 1.4367992e-01	  1.3092588e+00 	 2.0991302e-01


.. parsed-literal::

      96	 1.4168091e+00	 1.3796500e-01	 1.4704833e+00	 1.4335405e-01	  1.3064836e+00 	 2.0304990e-01


.. parsed-literal::

      97	 1.4201914e+00	 1.3790268e-01	 1.4737552e+00	 1.4311945e-01	[ 1.3107459e+00]	 2.0992541e-01


.. parsed-literal::

      98	 1.4230715e+00	 1.3777005e-01	 1.4766183e+00	 1.4302256e-01	[ 1.3148105e+00]	 2.1328449e-01
      99	 1.4272965e+00	 1.3747339e-01	 1.4809832e+00	 1.4267876e-01	[ 1.3209077e+00]	 1.8748641e-01


.. parsed-literal::

     100	 1.4299533e+00	 1.3752288e-01	 1.4839124e+00	 1.4324642e-01	  1.3042255e+00 	 2.0466828e-01
     101	 1.4348179e+00	 1.3718110e-01	 1.4886851e+00	 1.4257206e-01	  1.3169480e+00 	 1.9435573e-01


.. parsed-literal::

     102	 1.4372715e+00	 1.3701971e-01	 1.4911424e+00	 1.4232827e-01	  1.3170165e+00 	 1.9930792e-01
     103	 1.4405254e+00	 1.3676642e-01	 1.4944658e+00	 1.4206584e-01	  1.3122393e+00 	 2.0096397e-01


.. parsed-literal::

     104	 1.4459468e+00	 1.3619310e-01	 1.4999851e+00	 1.4160760e-01	  1.3055706e+00 	 2.1286654e-01


.. parsed-literal::

     105	 1.4490639e+00	 1.3560514e-01	 1.5034376e+00	 1.4101849e-01	  1.2945082e+00 	 2.0727372e-01


.. parsed-literal::

     106	 1.4530021e+00	 1.3554230e-01	 1.5072090e+00	 1.4110595e-01	  1.3025817e+00 	 2.0193768e-01


.. parsed-literal::

     107	 1.4553912e+00	 1.3549540e-01	 1.5095638e+00	 1.4106044e-01	  1.3067872e+00 	 2.0472002e-01


.. parsed-literal::

     108	 1.4583417e+00	 1.3536144e-01	 1.5125853e+00	 1.4086839e-01	  1.3115082e+00 	 2.0972133e-01


.. parsed-literal::

     109	 1.4614060e+00	 1.3541344e-01	 1.5158774e+00	 1.4054155e-01	  1.3035946e+00 	 2.0277476e-01


.. parsed-literal::

     110	 1.4652164e+00	 1.3532066e-01	 1.5196604e+00	 1.4032945e-01	  1.3086817e+00 	 2.0434546e-01


.. parsed-literal::

     111	 1.4674685e+00	 1.3527987e-01	 1.5218682e+00	 1.4019800e-01	  1.3087661e+00 	 2.1040797e-01
     112	 1.4710464e+00	 1.3513022e-01	 1.5254368e+00	 1.3983205e-01	  1.3068143e+00 	 1.9808555e-01


.. parsed-literal::

     113	 1.4732278e+00	 1.3518775e-01	 1.5276720e+00	 1.3978008e-01	  1.3049384e+00 	 3.1467342e-01


.. parsed-literal::

     114	 1.4762163e+00	 1.3505418e-01	 1.5306258e+00	 1.3942317e-01	  1.3036773e+00 	 2.1002960e-01


.. parsed-literal::

     115	 1.4789642e+00	 1.3490477e-01	 1.5333811e+00	 1.3909517e-01	  1.3028297e+00 	 2.0979261e-01
     116	 1.4817773e+00	 1.3468993e-01	 1.5362389e+00	 1.3857647e-01	  1.3051497e+00 	 1.8673897e-01


.. parsed-literal::

     117	 1.4835923e+00	 1.3460755e-01	 1.5381809e+00	 1.3839927e-01	  1.2980988e+00 	 2.1079946e-01


.. parsed-literal::

     118	 1.4857524e+00	 1.3441500e-01	 1.5402720e+00	 1.3824967e-01	  1.3019273e+00 	 2.2123528e-01
     119	 1.4878640e+00	 1.3414765e-01	 1.5423872e+00	 1.3800920e-01	  1.3025463e+00 	 1.9775224e-01


.. parsed-literal::

     120	 1.4897174e+00	 1.3380197e-01	 1.5442843e+00	 1.3779541e-01	  1.2976507e+00 	 1.9926953e-01
     121	 1.4916154e+00	 1.3307739e-01	 1.5463717e+00	 1.3747262e-01	  1.2952821e+00 	 1.9616890e-01


.. parsed-literal::

     122	 1.4946209e+00	 1.3282299e-01	 1.5493119e+00	 1.3736294e-01	  1.2892607e+00 	 2.1144247e-01


.. parsed-literal::

     123	 1.4959565e+00	 1.3268457e-01	 1.5506636e+00	 1.3735211e-01	  1.2887004e+00 	 2.1524763e-01


.. parsed-literal::

     124	 1.4977395e+00	 1.3243602e-01	 1.5525117e+00	 1.3735943e-01	  1.2903836e+00 	 2.1608543e-01


.. parsed-literal::

     125	 1.4998993e+00	 1.3209793e-01	 1.5548042e+00	 1.3734416e-01	  1.2929592e+00 	 2.1091247e-01
     126	 1.5022960e+00	 1.3184982e-01	 1.5572971e+00	 1.3727763e-01	  1.2989385e+00 	 1.9787216e-01


.. parsed-literal::

     127	 1.5039689e+00	 1.3176970e-01	 1.5589430e+00	 1.3717449e-01	  1.3023558e+00 	 2.1045661e-01


.. parsed-literal::

     128	 1.5059078e+00	 1.3158503e-01	 1.5608861e+00	 1.3697403e-01	  1.3050538e+00 	 2.1012855e-01
     129	 1.5073021e+00	 1.3146456e-01	 1.5623264e+00	 1.3683186e-01	  1.3038087e+00 	 1.8189454e-01


.. parsed-literal::

     130	 1.5088539e+00	 1.3129284e-01	 1.5638816e+00	 1.3678969e-01	  1.3036043e+00 	 2.0195127e-01
     131	 1.5105235e+00	 1.3097781e-01	 1.5656115e+00	 1.3666712e-01	  1.3026407e+00 	 1.9722152e-01


.. parsed-literal::

     132	 1.5117514e+00	 1.3082769e-01	 1.5668641e+00	 1.3666947e-01	  1.2999819e+00 	 2.1690583e-01


.. parsed-literal::

     133	 1.5138264e+00	 1.3042011e-01	 1.5690155e+00	 1.3652230e-01	  1.2981787e+00 	 2.1615434e-01
     134	 1.5151733e+00	 1.3014353e-01	 1.5704004e+00	 1.3655098e-01	  1.2967755e+00 	 2.0168948e-01


.. parsed-literal::

     135	 1.5166061e+00	 1.3007435e-01	 1.5717846e+00	 1.3647661e-01	  1.2976003e+00 	 2.0257115e-01
     136	 1.5179043e+00	 1.2996590e-01	 1.5730676e+00	 1.3637737e-01	  1.2981618e+00 	 2.0193005e-01


.. parsed-literal::

     137	 1.5193253e+00	 1.2983017e-01	 1.5745035e+00	 1.3628407e-01	  1.2979181e+00 	 1.9350624e-01


.. parsed-literal::

     138	 1.5216599e+00	 1.2967944e-01	 1.5769122e+00	 1.3625668e-01	  1.2930968e+00 	 2.0780849e-01


.. parsed-literal::

     139	 1.5228226e+00	 1.2953541e-01	 1.5781574e+00	 1.3623074e-01	  1.2903763e+00 	 2.1476936e-01
     140	 1.5245625e+00	 1.2958488e-01	 1.5798253e+00	 1.3629005e-01	  1.2900672e+00 	 1.9034863e-01


.. parsed-literal::

     141	 1.5256107e+00	 1.2962645e-01	 1.5808663e+00	 1.3635836e-01	  1.2871806e+00 	 2.1193504e-01


.. parsed-literal::

     142	 1.5269647e+00	 1.2957692e-01	 1.5822671e+00	 1.3637821e-01	  1.2805918e+00 	 2.0633817e-01


.. parsed-literal::

     143	 1.5279560e+00	 1.2944889e-01	 1.5834126e+00	 1.3632132e-01	  1.2653730e+00 	 2.1617317e-01


.. parsed-literal::

     144	 1.5297691e+00	 1.2935629e-01	 1.5851930e+00	 1.3627305e-01	  1.2645541e+00 	 2.1062064e-01
     145	 1.5307311e+00	 1.2923746e-01	 1.5861731e+00	 1.3616390e-01	  1.2631609e+00 	 1.9362068e-01


.. parsed-literal::

     146	 1.5317257e+00	 1.2911958e-01	 1.5872036e+00	 1.3605473e-01	  1.2599679e+00 	 2.0182920e-01


.. parsed-literal::

     147	 1.5335482e+00	 1.2890183e-01	 1.5890892e+00	 1.3582802e-01	  1.2556495e+00 	 2.0875454e-01


.. parsed-literal::

     148	 1.5345394e+00	 1.2874654e-01	 1.5901518e+00	 1.3577784e-01	  1.2496914e+00 	 3.0445337e-01


.. parsed-literal::

     149	 1.5359422e+00	 1.2856393e-01	 1.5915818e+00	 1.3558101e-01	  1.2465121e+00 	 2.0450568e-01
     150	 1.5370910e+00	 1.2836943e-01	 1.5927518e+00	 1.3543848e-01	  1.2450064e+00 	 1.8557405e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.17 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fe0b0625e10>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.07 s, sys: 51 ms, total: 2.12 s
    Wall time: 640 ms


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

