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
       1	-3.4392108e-01	 3.2053478e-01	-3.3418596e-01	 3.2031451e-01	[-3.3394182e-01]	 4.7385168e-01


.. parsed-literal::

       2	-2.7007485e-01	 3.0865494e-01	-2.4467056e-01	 3.1062248e-01	[-2.5083028e-01]	 2.2790909e-01


.. parsed-literal::

       3	-2.2666410e-01	 2.8867611e-01	-1.8368007e-01	 2.9297854e-01	[-2.0354487e-01]	 2.8173041e-01
       4	-1.9447056e-01	 2.6460016e-01	-1.5336656e-01	 2.6875840e-01	[-1.7861290e-01]	 1.9964147e-01


.. parsed-literal::

       5	-1.0385568e-01	 2.5664877e-01	-6.7671082e-02	 2.5881905e-01	[-7.9625409e-02]	 1.7619252e-01


.. parsed-literal::

       6	-6.9213934e-02	 2.5102166e-01	-3.7577386e-02	 2.5278523e-01	[-4.4609038e-02]	 2.1514440e-01


.. parsed-literal::

       7	-5.0894120e-02	 2.4829272e-01	-2.6239287e-02	 2.5000303e-01	[-3.2972764e-02]	 2.0333958e-01


.. parsed-literal::

       8	-3.9158535e-02	 2.4638801e-01	-1.8299496e-02	 2.4828832e-01	[-2.5927274e-02]	 2.0379472e-01


.. parsed-literal::

       9	-2.5383275e-02	 2.4387099e-01	-7.5927394e-03	 2.4631576e-01	[-1.7602361e-02]	 2.1243620e-01
      10	-1.3497100e-02	 2.4158160e-01	 2.0827326e-03	 2.4481620e-01	[-1.1925679e-02]	 2.0365882e-01


.. parsed-literal::

      11	-1.0366431e-02	 2.4136627e-01	 3.8527188e-03	 2.4429332e-01	[-8.2672142e-03]	 2.0556998e-01
      12	-6.3049109e-03	 2.4051479e-01	 7.6202858e-03	 2.4385085e-01	[-6.2776210e-03]	 1.9884086e-01


.. parsed-literal::

      13	-3.3335912e-03	 2.3989067e-01	 1.0548383e-02	 2.4344246e-01	[-4.3800884e-03]	 1.9994211e-01


.. parsed-literal::

      14	 5.7145098e-04	 2.3907097e-01	 1.4779674e-02	 2.4273349e-01	[-9.6455377e-04]	 2.0372629e-01


.. parsed-literal::

      15	 8.3211280e-02	 2.2536841e-01	 1.0241324e-01	 2.2607923e-01	[ 9.7139695e-02]	 2.9989505e-01


.. parsed-literal::

      16	 1.8004429e-01	 2.2140343e-01	 2.0241443e-01	 2.2576579e-01	[ 1.8636459e-01]	 2.1494699e-01


.. parsed-literal::

      17	 2.7755498e-01	 2.1830417e-01	 3.0997454e-01	 2.2556610e-01	[ 2.7161239e-01]	 2.0779276e-01


.. parsed-literal::

      18	 3.1856225e-01	 2.1735786e-01	 3.5022340e-01	 2.2697538e-01	[ 3.2083997e-01]	 2.1060395e-01
      19	 3.5843474e-01	 2.1170833e-01	 3.9077196e-01	 2.1878884e-01	[ 3.7208705e-01]	 2.0138288e-01


.. parsed-literal::

      20	 4.0377482e-01	 2.0799429e-01	 4.3664839e-01	 2.1362569e-01	[ 4.2117947e-01]	 1.9895315e-01
      21	 4.5888246e-01	 2.0588297e-01	 4.9322863e-01	 2.1198014e-01	[ 4.7506453e-01]	 1.9751239e-01


.. parsed-literal::

      22	 5.3187897e-01	 2.0458569e-01	 5.6866948e-01	 2.1187056e-01	[ 5.5398435e-01]	 2.0752072e-01


.. parsed-literal::

      23	 5.7602430e-01	 2.0432695e-01	 6.1386129e-01	 2.1151229e-01	[ 6.0978977e-01]	 2.0840001e-01


.. parsed-literal::

      24	 6.0912272e-01	 2.0108743e-01	 6.4914601e-01	 2.0753504e-01	[ 6.3737961e-01]	 2.0984578e-01


.. parsed-literal::

      25	 6.4327993e-01	 2.0025520e-01	 6.8171976e-01	 2.0760232e-01	[ 6.6236770e-01]	 2.0393634e-01
      26	 6.7605631e-01	 1.9848327e-01	 7.1423696e-01	 2.0646904e-01	[ 6.8922705e-01]	 1.8224478e-01


.. parsed-literal::

      27	 7.3309080e-01	 2.0244801e-01	 7.7080463e-01	 2.0773765e-01	[ 7.3471111e-01]	 2.0005560e-01
      28	 7.7048615e-01	 2.0568264e-01	 8.0893276e-01	 2.1222658e-01	[ 7.6469673e-01]	 1.7258501e-01


.. parsed-literal::

      29	 8.0579298e-01	 2.0068893e-01	 8.4696585e-01	 2.0710724e-01	[ 7.9740813e-01]	 2.0445132e-01
      30	 8.3250214e-01	 2.0205875e-01	 8.7431186e-01	 2.0839994e-01	[ 8.1722912e-01]	 1.7665672e-01


.. parsed-literal::

      31	 8.5370211e-01	 2.0161189e-01	 8.9606532e-01	 2.0799067e-01	[ 8.3494796e-01]	 2.1764517e-01


.. parsed-literal::

      32	 8.8155246e-01	 2.0007273e-01	 9.2501605e-01	 2.0672597e-01	[ 8.6452459e-01]	 2.1386337e-01


.. parsed-literal::

      33	 9.0251776e-01	 1.9758425e-01	 9.4613878e-01	 2.0442328e-01	[ 8.7131147e-01]	 2.1113706e-01


.. parsed-literal::

      34	 9.2187307e-01	 1.9679444e-01	 9.6579453e-01	 2.0399730e-01	[ 8.9680559e-01]	 2.1379638e-01


.. parsed-literal::

      35	 9.3795477e-01	 1.9642815e-01	 9.8244869e-01	 2.0352265e-01	[ 9.1880163e-01]	 2.1375799e-01
      36	 9.5788993e-01	 1.9460120e-01	 1.0036159e+00	 2.0118842e-01	[ 9.3728043e-01]	 1.9096994e-01


.. parsed-literal::

      37	 9.7649984e-01	 1.9384008e-01	 1.0221960e+00	 2.0011611e-01	[ 9.5005565e-01]	 2.0687151e-01
      38	 9.9094581e-01	 1.8804670e-01	 1.0382901e+00	 1.9498690e-01	[ 9.6077234e-01]	 2.0208263e-01


.. parsed-literal::

      39	 1.0044018e+00	 1.8490161e-01	 1.0515027e+00	 1.9192061e-01	[ 9.7525861e-01]	 1.9878078e-01
      40	 1.0167405e+00	 1.8141108e-01	 1.0636534e+00	 1.8823144e-01	[ 9.8873289e-01]	 1.7872906e-01


.. parsed-literal::

      41	 1.0343237e+00	 1.7714468e-01	 1.0813658e+00	 1.8357478e-01	[ 1.0044850e+00]	 2.0410347e-01
      42	 1.0528781e+00	 1.7332231e-01	 1.1001361e+00	 1.7987024e-01	[ 1.0186219e+00]	 2.0232201e-01


.. parsed-literal::

      43	 1.0681298e+00	 1.7083419e-01	 1.1165675e+00	 1.7689161e-01	[ 1.0237640e+00]	 2.0354295e-01
      44	 1.0841878e+00	 1.6953639e-01	 1.1323371e+00	 1.7678775e-01	[ 1.0380340e+00]	 1.7708039e-01


.. parsed-literal::

      45	 1.1009367e+00	 1.6968150e-01	 1.1494345e+00	 1.7785651e-01	[ 1.0425940e+00]	 1.9730186e-01
      46	 1.1133993e+00	 1.7018417e-01	 1.1618615e+00	 1.7877545e-01	[ 1.0456419e+00]	 2.0529747e-01


.. parsed-literal::

      47	 1.1285419e+00	 1.6803977e-01	 1.1771353e+00	 1.7528890e-01	[ 1.0490573e+00]	 2.0049238e-01
      48	 1.1437910e+00	 1.6535171e-01	 1.1926023e+00	 1.7197682e-01	[ 1.0566319e+00]	 1.7120123e-01


.. parsed-literal::

      49	 1.1656460e+00	 1.6077631e-01	 1.2150354e+00	 1.6608765e-01	[ 1.0605602e+00]	 2.0741725e-01


.. parsed-literal::

      50	 1.1789837e+00	 1.5831202e-01	 1.2284001e+00	 1.6382104e-01	  1.0596266e+00 	 2.1309018e-01


.. parsed-literal::

      51	 1.1909766e+00	 1.5675870e-01	 1.2404350e+00	 1.6243880e-01	  1.0543941e+00 	 2.0726848e-01
      52	 1.2040261e+00	 1.5481012e-01	 1.2538891e+00	 1.6044781e-01	  1.0300407e+00 	 1.7680860e-01


.. parsed-literal::

      53	 1.2159622e+00	 1.5266529e-01	 1.2657839e+00	 1.5786731e-01	  1.0172422e+00 	 2.0235872e-01
      54	 1.2302744e+00	 1.4945591e-01	 1.2808055e+00	 1.5384580e-01	  1.0011208e+00 	 1.8266892e-01


.. parsed-literal::

      55	 1.2423117e+00	 1.4776071e-01	 1.2928588e+00	 1.5200753e-01	  1.0251433e+00 	 1.7125058e-01


.. parsed-literal::

      56	 1.2554472e+00	 1.4614733e-01	 1.3059150e+00	 1.5036549e-01	[ 1.0635053e+00]	 2.0438170e-01


.. parsed-literal::

      57	 1.2683098e+00	 1.4427147e-01	 1.3193006e+00	 1.4848393e-01	[ 1.0852001e+00]	 2.1200442e-01
      58	 1.2801199e+00	 1.4373546e-01	 1.3310658e+00	 1.4792100e-01	[ 1.1147732e+00]	 2.0787787e-01


.. parsed-literal::

      59	 1.2881304e+00	 1.4330287e-01	 1.3390311e+00	 1.4737548e-01	[ 1.1254992e+00]	 2.0345092e-01


.. parsed-literal::

      60	 1.2986194e+00	 1.4312303e-01	 1.3499526e+00	 1.4729712e-01	[ 1.1367927e+00]	 2.0409417e-01
      61	 1.3065646e+00	 1.4201501e-01	 1.3580039e+00	 1.4570897e-01	[ 1.1613281e+00]	 1.9834876e-01


.. parsed-literal::

      62	 1.3143821e+00	 1.4202875e-01	 1.3655763e+00	 1.4579601e-01	[ 1.1739410e+00]	 2.1433210e-01


.. parsed-literal::

      63	 1.3252657e+00	 1.4070513e-01	 1.3766046e+00	 1.4410787e-01	[ 1.1938992e+00]	 2.1693349e-01


.. parsed-literal::

      64	 1.3344328e+00	 1.4033080e-01	 1.3861025e+00	 1.4405765e-01	[ 1.2003149e+00]	 2.0920992e-01
      65	 1.3453089e+00	 1.3873406e-01	 1.3968644e+00	 1.4232262e-01	[ 1.2065961e+00]	 2.1053123e-01


.. parsed-literal::

      66	 1.3529488e+00	 1.3738971e-01	 1.4048655e+00	 1.4145566e-01	  1.1876640e+00 	 2.0554662e-01
      67	 1.3594133e+00	 1.3708665e-01	 1.4113528e+00	 1.4125581e-01	  1.1853794e+00 	 1.8335533e-01


.. parsed-literal::

      68	 1.3651498e+00	 1.3684872e-01	 1.4171153e+00	 1.4124500e-01	  1.1764144e+00 	 2.1833587e-01


.. parsed-literal::

      69	 1.3698537e+00	 1.3632499e-01	 1.4219668e+00	 1.4102154e-01	  1.1512088e+00 	 2.0310163e-01


.. parsed-literal::

      70	 1.3767532e+00	 1.3565095e-01	 1.4288043e+00	 1.4061319e-01	  1.1453886e+00 	 2.0560551e-01
      71	 1.3812940e+00	 1.3509030e-01	 1.4334404e+00	 1.4052413e-01	  1.1307769e+00 	 1.7933440e-01


.. parsed-literal::

      72	 1.3862835e+00	 1.3439580e-01	 1.4386110e+00	 1.4033555e-01	  1.1146103e+00 	 2.0398641e-01
      73	 1.3930247e+00	 1.3366258e-01	 1.4456575e+00	 1.4097026e-01	  1.0741679e+00 	 1.8539739e-01


.. parsed-literal::

      74	 1.3994900e+00	 1.3304197e-01	 1.4521298e+00	 1.4071988e-01	  1.0631845e+00 	 2.1381259e-01


.. parsed-literal::

      75	 1.4040561e+00	 1.3262025e-01	 1.4566559e+00	 1.4045528e-01	  1.0642202e+00 	 2.0524812e-01
      76	 1.4113784e+00	 1.3200955e-01	 1.4642110e+00	 1.4031610e-01	  1.0589996e+00 	 1.8695307e-01


.. parsed-literal::

      77	 1.4162137e+00	 1.3085559e-01	 1.4692137e+00	 1.3970621e-01	  1.0499840e+00 	 2.0506048e-01


.. parsed-literal::

      78	 1.4223520e+00	 1.3081490e-01	 1.4754224e+00	 1.3992082e-01	  1.0544542e+00 	 2.1790242e-01


.. parsed-literal::

      79	 1.4261530e+00	 1.3062328e-01	 1.4793725e+00	 1.4001227e-01	  1.0500874e+00 	 2.1426010e-01


.. parsed-literal::

      80	 1.4305289e+00	 1.3039865e-01	 1.4839726e+00	 1.3995757e-01	  1.0480905e+00 	 2.0795989e-01


.. parsed-literal::

      81	 1.4349746e+00	 1.2973847e-01	 1.4885571e+00	 1.3960424e-01	  1.0449624e+00 	 2.0258904e-01


.. parsed-literal::

      82	 1.4397552e+00	 1.2950081e-01	 1.4933013e+00	 1.3955176e-01	  1.0466560e+00 	 2.1831083e-01


.. parsed-literal::

      83	 1.4429445e+00	 1.2941664e-01	 1.4964079e+00	 1.3947732e-01	  1.0487145e+00 	 2.0782614e-01


.. parsed-literal::

      84	 1.4479202e+00	 1.2908820e-01	 1.5014319e+00	 1.3929424e-01	  1.0460867e+00 	 2.1546841e-01


.. parsed-literal::

      85	 1.4489563e+00	 1.3004708e-01	 1.5026157e+00	 1.4044027e-01	  1.0376815e+00 	 2.1059918e-01


.. parsed-literal::

      86	 1.4556939e+00	 1.2889473e-01	 1.5092204e+00	 1.3941202e-01	  1.0496621e+00 	 2.0671129e-01
      87	 1.4579006e+00	 1.2853197e-01	 1.5114554e+00	 1.3915199e-01	  1.0512849e+00 	 1.7812395e-01


.. parsed-literal::

      88	 1.4614645e+00	 1.2814390e-01	 1.5151197e+00	 1.3891628e-01	  1.0484826e+00 	 2.0870662e-01
      89	 1.4663143e+00	 1.2777227e-01	 1.5200784e+00	 1.3876935e-01	  1.0364099e+00 	 1.9473028e-01


.. parsed-literal::

      90	 1.4691156e+00	 1.2707727e-01	 1.5230126e+00	 1.3812840e-01	  1.0297745e+00 	 3.1444025e-01
      91	 1.4727303e+00	 1.2697860e-01	 1.5266494e+00	 1.3811384e-01	  1.0213367e+00 	 1.9560385e-01


.. parsed-literal::

      92	 1.4752580e+00	 1.2674242e-01	 1.5291746e+00	 1.3790036e-01	  1.0166036e+00 	 1.7837787e-01


.. parsed-literal::

      93	 1.4793984e+00	 1.2635554e-01	 1.5334086e+00	 1.3752051e-01	  1.0198165e+00 	 2.1298218e-01


.. parsed-literal::

      94	 1.4814287e+00	 1.2534056e-01	 1.5356248e+00	 1.3646325e-01	  1.0070350e+00 	 2.1144295e-01


.. parsed-literal::

      95	 1.4848340e+00	 1.2526913e-01	 1.5389355e+00	 1.3643962e-01	  1.0249984e+00 	 2.1250844e-01


.. parsed-literal::

      96	 1.4869822e+00	 1.2505014e-01	 1.5411094e+00	 1.3624217e-01	  1.0354727e+00 	 2.0456958e-01


.. parsed-literal::

      97	 1.4896414e+00	 1.2471345e-01	 1.5438668e+00	 1.3596434e-01	  1.0438903e+00 	 2.2123051e-01


.. parsed-literal::

      98	 1.4920842e+00	 1.2413456e-01	 1.5464800e+00	 1.3548044e-01	  1.0536661e+00 	 2.0220494e-01


.. parsed-literal::

      99	 1.4949982e+00	 1.2405194e-01	 1.5492926e+00	 1.3550545e-01	  1.0558932e+00 	 2.1238875e-01
     100	 1.4967292e+00	 1.2387895e-01	 1.5509693e+00	 1.3547465e-01	  1.0581511e+00 	 1.9957209e-01


.. parsed-literal::

     101	 1.4985689e+00	 1.2364674e-01	 1.5527559e+00	 1.3543353e-01	  1.0610610e+00 	 2.1234393e-01


.. parsed-literal::

     102	 1.5008992e+00	 1.2283901e-01	 1.5550856e+00	 1.3501204e-01	  1.0745830e+00 	 2.0695233e-01


.. parsed-literal::

     103	 1.5036574e+00	 1.2252095e-01	 1.5578318e+00	 1.3478689e-01	  1.0795582e+00 	 2.1463776e-01


.. parsed-literal::

     104	 1.5055208e+00	 1.2224150e-01	 1.5597483e+00	 1.3452579e-01	  1.0836558e+00 	 2.0713830e-01
     105	 1.5082808e+00	 1.2175355e-01	 1.5626331e+00	 1.3402256e-01	  1.0878103e+00 	 1.7261028e-01


.. parsed-literal::

     106	 1.5107394e+00	 1.2156251e-01	 1.5653164e+00	 1.3398468e-01	  1.0874595e+00 	 1.9416046e-01
     107	 1.5140369e+00	 1.2104502e-01	 1.5686440e+00	 1.3333921e-01	  1.0884243e+00 	 1.9767833e-01


.. parsed-literal::

     108	 1.5161881e+00	 1.2088576e-01	 1.5707522e+00	 1.3317787e-01	  1.0883877e+00 	 2.0845246e-01
     109	 1.5186237e+00	 1.2078478e-01	 1.5732066e+00	 1.3316333e-01	  1.0818751e+00 	 2.0451880e-01


.. parsed-literal::

     110	 1.5197009e+00	 1.2044458e-01	 1.5744599e+00	 1.3282691e-01	  1.0803665e+00 	 2.0149779e-01
     111	 1.5223102e+00	 1.2051027e-01	 1.5769908e+00	 1.3295410e-01	  1.0750436e+00 	 1.9499922e-01


.. parsed-literal::

     112	 1.5239844e+00	 1.2043949e-01	 1.5787155e+00	 1.3293011e-01	  1.0697202e+00 	 2.0882940e-01


.. parsed-literal::

     113	 1.5257239e+00	 1.2029543e-01	 1.5805597e+00	 1.3285943e-01	  1.0629552e+00 	 2.1625352e-01


.. parsed-literal::

     114	 1.5285321e+00	 1.2001741e-01	 1.5835074e+00	 1.3275499e-01	  1.0533593e+00 	 2.1035719e-01
     115	 1.5292804e+00	 1.1978409e-01	 1.5845294e+00	 1.3291614e-01	  1.0403237e+00 	 1.9561577e-01


.. parsed-literal::

     116	 1.5324857e+00	 1.1964570e-01	 1.5875566e+00	 1.3275619e-01	  1.0476578e+00 	 2.0028377e-01


.. parsed-literal::

     117	 1.5334905e+00	 1.1955553e-01	 1.5885037e+00	 1.3274955e-01	  1.0505673e+00 	 2.1692729e-01
     118	 1.5353568e+00	 1.1937604e-01	 1.5903953e+00	 1.3284684e-01	  1.0504351e+00 	 1.9830227e-01


.. parsed-literal::

     119	 1.5364626e+00	 1.1935660e-01	 1.5916615e+00	 1.3347926e-01	  1.0422049e+00 	 1.8431449e-01


.. parsed-literal::

     120	 1.5386256e+00	 1.1923949e-01	 1.5938173e+00	 1.3334000e-01	  1.0415489e+00 	 2.1975231e-01


.. parsed-literal::

     121	 1.5398759e+00	 1.1922809e-01	 1.5951430e+00	 1.3339459e-01	  1.0359926e+00 	 2.0509315e-01
     122	 1.5410404e+00	 1.1920449e-01	 1.5963750e+00	 1.3344668e-01	  1.0303816e+00 	 1.9810939e-01


.. parsed-literal::

     123	 1.5433559e+00	 1.1918830e-01	 1.5988023e+00	 1.3359014e-01	  1.0219712e+00 	 1.9993281e-01


.. parsed-literal::

     124	 1.5444802e+00	 1.1915401e-01	 1.5999541e+00	 1.3371299e-01	  1.0182999e+00 	 3.2310653e-01


.. parsed-literal::

     125	 1.5462973e+00	 1.1917726e-01	 1.6017817e+00	 1.3386740e-01	  1.0198903e+00 	 2.1051359e-01
     126	 1.5475216e+00	 1.1916807e-01	 1.6029461e+00	 1.3390751e-01	  1.0232904e+00 	 1.9589615e-01


.. parsed-literal::

     127	 1.5491403e+00	 1.1912223e-01	 1.6045021e+00	 1.3397545e-01	  1.0298375e+00 	 2.0691061e-01


.. parsed-literal::

     128	 1.5500783e+00	 1.1918851e-01	 1.6055761e+00	 1.3438147e-01	  1.0270392e+00 	 2.1068072e-01
     129	 1.5517198e+00	 1.1910686e-01	 1.6071713e+00	 1.3431019e-01	  1.0292410e+00 	 1.8056130e-01


.. parsed-literal::

     130	 1.5532926e+00	 1.1911576e-01	 1.6088485e+00	 1.3450529e-01	  1.0274748e+00 	 2.1557546e-01


.. parsed-literal::

     131	 1.5548605e+00	 1.1912457e-01	 1.6105512e+00	 1.3476830e-01	  1.0237704e+00 	 2.0943737e-01
     132	 1.5569389e+00	 1.1925449e-01	 1.6127927e+00	 1.3530415e-01	  1.0234133e+00 	 1.9869137e-01


.. parsed-literal::

     133	 1.5583655e+00	 1.1909314e-01	 1.6142303e+00	 1.3554189e-01	  1.0250034e+00 	 2.0899606e-01


.. parsed-literal::

     134	 1.5594236e+00	 1.1901033e-01	 1.6151596e+00	 1.3527237e-01	  1.0268854e+00 	 2.1276402e-01
     135	 1.5602826e+00	 1.1900483e-01	 1.6159390e+00	 1.3525404e-01	  1.0327260e+00 	 1.7908430e-01


.. parsed-literal::

     136	 1.5616492e+00	 1.1896111e-01	 1.6172500e+00	 1.3525933e-01	  1.0375651e+00 	 1.7498636e-01


.. parsed-literal::

     137	 1.5632017e+00	 1.1896999e-01	 1.6189008e+00	 1.3540927e-01	  1.0536634e+00 	 2.0426059e-01
     138	 1.5652107e+00	 1.1896548e-01	 1.6208756e+00	 1.3552494e-01	  1.0500589e+00 	 1.9721627e-01


.. parsed-literal::

     139	 1.5661692e+00	 1.1893601e-01	 1.6219080e+00	 1.3556430e-01	  1.0449486e+00 	 2.0004845e-01
     140	 1.5672608e+00	 1.1893079e-01	 1.6231209e+00	 1.3564705e-01	  1.0390192e+00 	 1.7250133e-01


.. parsed-literal::

     141	 1.5681162e+00	 1.1890529e-01	 1.6240664e+00	 1.3560504e-01	  1.0329335e+00 	 3.2532597e-01
     142	 1.5692490e+00	 1.1891468e-01	 1.6252449e+00	 1.3566066e-01	  1.0308714e+00 	 1.9536328e-01


.. parsed-literal::

     143	 1.5703756e+00	 1.1890730e-01	 1.6263572e+00	 1.3559756e-01	  1.0303752e+00 	 1.8213081e-01


.. parsed-literal::

     144	 1.5713926e+00	 1.1891836e-01	 1.6273229e+00	 1.3556173e-01	  1.0321706e+00 	 2.0918918e-01


.. parsed-literal::

     145	 1.5726536e+00	 1.1893066e-01	 1.6284995e+00	 1.3544720e-01	  1.0343934e+00 	 2.0971656e-01
     146	 1.5738692e+00	 1.1898484e-01	 1.6296431e+00	 1.3549082e-01	  1.0381012e+00 	 2.0071054e-01


.. parsed-literal::

     147	 1.5748688e+00	 1.1901555e-01	 1.6306134e+00	 1.3555338e-01	  1.0377043e+00 	 2.0118403e-01
     148	 1.5761369e+00	 1.1909618e-01	 1.6319300e+00	 1.3568772e-01	  1.0341484e+00 	 1.9964767e-01


.. parsed-literal::

     149	 1.5773161e+00	 1.1917863e-01	 1.6332328e+00	 1.3579729e-01	  1.0286557e+00 	 2.0620728e-01


.. parsed-literal::

     150	 1.5786576e+00	 1.1919116e-01	 1.6347960e+00	 1.3586216e-01	  1.0206765e+00 	 2.1043539e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.13 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f69540b5810>



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
    CPU times: user 1.96 s, sys: 53.9 ms, total: 2.02 s
    Wall time: 595 ms


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

