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
       1	-3.4418614e-01	 3.2092617e-01	-3.3444763e-01	 3.1958291e-01	[-3.3121612e-01]	 4.6894932e-01


.. parsed-literal::

       2	-2.7455343e-01	 3.1058661e-01	-2.5086551e-01	 3.0880932e-01	[-2.4531565e-01]	 2.2805119e-01


.. parsed-literal::

       3	-2.3197084e-01	 2.9080127e-01	-1.9087865e-01	 2.8813471e-01	[-1.8185699e-01]	 2.7569675e-01
       4	-1.9181081e-01	 2.6649872e-01	-1.4825039e-01	 2.6690283e-01	[-1.5076869e-01]	 1.7701411e-01


.. parsed-literal::

       5	-1.0985743e-01	 2.5735132e-01	-7.6320897e-02	 2.5832837e-01	[-8.3698650e-02]	 2.0623565e-01


.. parsed-literal::

       6	-6.9905817e-02	 2.5129537e-01	-3.9318603e-02	 2.5373243e-01	[-4.9132444e-02]	 2.0911217e-01


.. parsed-literal::

       7	-5.3795045e-02	 2.4891940e-01	-2.8560659e-02	 2.5178413e-01	[-4.0550947e-02]	 2.0721531e-01
       8	-3.7054340e-02	 2.4603704e-01	-1.6478536e-02	 2.4923523e-01	[-3.0109255e-02]	 1.8559790e-01


.. parsed-literal::

       9	-2.4794802e-02	 2.4379820e-01	-7.2024430e-03	 2.4633765e-01	[-1.8357028e-02]	 2.0893979e-01


.. parsed-literal::

      10	-1.6724364e-02	 2.4234904e-01	-1.5052375e-03	 2.4326629e-01	[-5.6938697e-03]	 2.0372963e-01


.. parsed-literal::

      11	-8.9684373e-03	 2.4106123e-01	 5.4121749e-03	 2.4146119e-01	[ 3.3724598e-03]	 2.1020865e-01


.. parsed-literal::

      12	-6.5664585e-03	 2.4066940e-01	 7.4616551e-03	 2.4154024e-01	[ 4.1752286e-03]	 2.1183634e-01


.. parsed-literal::

      13	-3.4328258e-03	 2.4014875e-01	 1.0342356e-02	 2.4124295e-01	[ 5.8827639e-03]	 2.1364355e-01


.. parsed-literal::

      14	 5.7648816e-02	 2.2973048e-01	 7.4853515e-02	 2.3280631e-01	[ 6.4486298e-02]	 3.1410289e-01


.. parsed-literal::

      15	 8.3557202e-02	 2.2499663e-01	 1.0367105e-01	 2.2757688e-01	[ 9.4878248e-02]	 3.2757616e-01


.. parsed-literal::

      16	 1.4268428e-01	 2.1815554e-01	 1.6471899e-01	 2.2036542e-01	[ 1.5659600e-01]	 2.1319103e-01
      17	 2.6943751e-01	 2.1979384e-01	 2.9766916e-01	 2.2585528e-01	[ 2.7875404e-01]	 1.7269015e-01


.. parsed-literal::

      18	 3.2899174e-01	 2.1636146e-01	 3.5916880e-01	 2.1940636e-01	[ 3.4004385e-01]	 2.0047998e-01
      19	 3.9999720e-01	 2.1008442e-01	 4.3202393e-01	 2.1200291e-01	[ 4.1357669e-01]	 1.9974041e-01


.. parsed-literal::

      20	 4.8148435e-01	 2.0553026e-01	 5.1513280e-01	 2.0659642e-01	[ 5.0072134e-01]	 2.0134306e-01
      21	 5.7118933e-01	 2.0025340e-01	 6.0741063e-01	 2.0087325e-01	[ 6.0165082e-01]	 1.9928885e-01


.. parsed-literal::

      22	 6.2997631e-01	 1.9255008e-01	 6.6885860e-01	 1.9433172e-01	[ 6.6279254e-01]	 2.0947456e-01


.. parsed-literal::

      23	 6.7719465e-01	 1.8548139e-01	 7.1628467e-01	 1.8849852e-01	[ 7.1250278e-01]	 2.1394515e-01


.. parsed-literal::

      24	 7.1576502e-01	 1.8293058e-01	 7.5448897e-01	 1.8725499e-01	[ 7.4423838e-01]	 2.0747113e-01


.. parsed-literal::

      25	 7.5568281e-01	 1.8192411e-01	 7.9351943e-01	 1.8694806e-01	[ 7.7803834e-01]	 2.0622110e-01
      26	 7.9177014e-01	 1.8442217e-01	 8.2962087e-01	 1.8854732e-01	[ 8.0778627e-01]	 1.9835830e-01


.. parsed-literal::

      27	 8.2530762e-01	 1.8434149e-01	 8.6391098e-01	 1.8903728e-01	[ 8.4126379e-01]	 2.0235968e-01


.. parsed-literal::

      28	 8.4772548e-01	 1.8330620e-01	 8.8833150e-01	 1.9031635e-01	[ 8.5380572e-01]	 2.0783949e-01


.. parsed-literal::

      29	 8.7081231e-01	 1.7903033e-01	 9.1100852e-01	 1.8525491e-01	[ 8.7683744e-01]	 2.1092534e-01
      30	 8.8246584e-01	 1.7669165e-01	 9.2266454e-01	 1.8275243e-01	[ 8.8591873e-01]	 1.9262123e-01


.. parsed-literal::

      31	 9.0181002e-01	 1.7328726e-01	 9.4229710e-01	 1.7913361e-01	[ 9.0043932e-01]	 1.9371104e-01
      32	 9.2009677e-01	 1.7140973e-01	 9.6118936e-01	 1.7645936e-01	[ 9.2056151e-01]	 1.9849515e-01


.. parsed-literal::

      33	 9.3767966e-01	 1.6893125e-01	 9.7938312e-01	 1.7393692e-01	[ 9.3591972e-01]	 1.8186450e-01


.. parsed-literal::

      34	 9.5333776e-01	 1.6899692e-01	 9.9544810e-01	 1.7366851e-01	[ 9.5379602e-01]	 2.1313715e-01
      35	 9.6803087e-01	 1.6741656e-01	 1.0104949e+00	 1.7212497e-01	[ 9.6866889e-01]	 1.8240619e-01


.. parsed-literal::

      36	 9.8530400e-01	 1.6596864e-01	 1.0283216e+00	 1.7052781e-01	[ 9.8145906e-01]	 2.0174336e-01


.. parsed-literal::

      37	 1.0003346e+00	 1.6489255e-01	 1.0436204e+00	 1.6837126e-01	[ 9.9079799e-01]	 2.0722508e-01
      38	 1.0164081e+00	 1.6473439e-01	 1.0603304e+00	 1.6603323e-01	[ 9.9660261e-01]	 1.9411254e-01


.. parsed-literal::

      39	 1.0291297e+00	 1.6347796e-01	 1.0732584e+00	 1.6378869e-01	[ 1.0076532e+00]	 1.7137790e-01


.. parsed-literal::

      40	 1.0373496e+00	 1.6288343e-01	 1.0815941e+00	 1.6312209e-01	[ 1.0163046e+00]	 2.0704103e-01


.. parsed-literal::

      41	 1.0481427e+00	 1.6196158e-01	 1.0927462e+00	 1.6209610e-01	[ 1.0289839e+00]	 2.0364189e-01


.. parsed-literal::

      42	 1.0592354e+00	 1.6049346e-01	 1.1047474e+00	 1.5964173e-01	[ 1.0406877e+00]	 2.1193242e-01
      43	 1.0706809e+00	 1.5915792e-01	 1.1167872e+00	 1.5874066e-01	[ 1.0485810e+00]	 1.9780660e-01


.. parsed-literal::

      44	 1.0799093e+00	 1.5835389e-01	 1.1262341e+00	 1.5840359e-01	[ 1.0514962e+00]	 1.9980907e-01
      45	 1.0924517e+00	 1.5731845e-01	 1.1392051e+00	 1.5814065e-01	  1.0507897e+00 	 1.9864559e-01


.. parsed-literal::

      46	 1.1023272e+00	 1.5721746e-01	 1.1492547e+00	 1.5827744e-01	[ 1.0518192e+00]	 1.7874575e-01


.. parsed-literal::

      47	 1.1107471e+00	 1.5650077e-01	 1.1576125e+00	 1.5825817e-01	[ 1.0567292e+00]	 2.1071053e-01


.. parsed-literal::

      48	 1.1202689e+00	 1.5542432e-01	 1.1668027e+00	 1.5829427e-01	[ 1.0613431e+00]	 2.0594621e-01


.. parsed-literal::

      49	 1.1277545e+00	 1.5444459e-01	 1.1745971e+00	 1.5768182e-01	[ 1.0718546e+00]	 2.0629478e-01
      50	 1.1370838e+00	 1.5304481e-01	 1.1839162e+00	 1.5593203e-01	[ 1.0746725e+00]	 2.0253825e-01


.. parsed-literal::

      51	 1.1444491e+00	 1.5162698e-01	 1.1915716e+00	 1.5488211e-01	  1.0729057e+00 	 2.1433759e-01
      52	 1.1562024e+00	 1.5001700e-01	 1.2036959e+00	 1.5334885e-01	[ 1.0784115e+00]	 1.8500376e-01


.. parsed-literal::

      53	 1.1658809e+00	 1.4862307e-01	 1.2138517e+00	 1.5234756e-01	[ 1.0920954e+00]	 2.0546818e-01


.. parsed-literal::

      54	 1.1757433e+00	 1.4815692e-01	 1.2235878e+00	 1.5195290e-01	[ 1.1073288e+00]	 2.0949864e-01
      55	 1.1826726e+00	 1.4766063e-01	 1.2303646e+00	 1.5194633e-01	[ 1.1151208e+00]	 1.7565203e-01


.. parsed-literal::

      56	 1.1908869e+00	 1.4659234e-01	 1.2388699e+00	 1.5256744e-01	[ 1.1233725e+00]	 2.0549679e-01
      57	 1.1988326e+00	 1.4585720e-01	 1.2470750e+00	 1.5378991e-01	  1.1218552e+00 	 1.8613791e-01


.. parsed-literal::

      58	 1.2052447e+00	 1.4516741e-01	 1.2535646e+00	 1.5370877e-01	  1.1214152e+00 	 2.0070744e-01
      59	 1.2156250e+00	 1.4409946e-01	 1.2643859e+00	 1.5338692e-01	  1.1212716e+00 	 1.8427777e-01


.. parsed-literal::

      60	 1.2241727e+00	 1.4371179e-01	 1.2732260e+00	 1.5227951e-01	  1.1183330e+00 	 2.0437551e-01
      61	 1.2337804e+00	 1.4311885e-01	 1.2832505e+00	 1.5141141e-01	  1.1117045e+00 	 1.8514156e-01


.. parsed-literal::

      62	 1.2398543e+00	 1.4287673e-01	 1.2891993e+00	 1.5042396e-01	  1.1153696e+00 	 2.0823598e-01
      63	 1.2449352e+00	 1.4268148e-01	 1.2942975e+00	 1.5047738e-01	  1.1198970e+00 	 1.9732833e-01


.. parsed-literal::

      64	 1.2520071e+00	 1.4200181e-01	 1.3015004e+00	 1.5064439e-01	  1.1212691e+00 	 2.0408106e-01


.. parsed-literal::

      65	 1.2591182e+00	 1.4172387e-01	 1.3090686e+00	 1.5117701e-01	[ 1.1323178e+00]	 2.1494555e-01


.. parsed-literal::

      66	 1.2666029e+00	 1.4091301e-01	 1.3165999e+00	 1.5096092e-01	  1.1271205e+00 	 2.0385695e-01
      67	 1.2700532e+00	 1.4052167e-01	 1.3200540e+00	 1.5044102e-01	  1.1288720e+00 	 1.9092107e-01


.. parsed-literal::

      68	 1.2779403e+00	 1.3991644e-01	 1.3281320e+00	 1.4983914e-01	[ 1.1323448e+00]	 2.1048474e-01
      69	 1.2836198e+00	 1.3984913e-01	 1.3339423e+00	 1.4886674e-01	[ 1.1389665e+00]	 1.9672799e-01


.. parsed-literal::

      70	 1.2901684e+00	 1.3956755e-01	 1.3405180e+00	 1.4888493e-01	[ 1.1444053e+00]	 2.0249248e-01


.. parsed-literal::

      71	 1.2966050e+00	 1.3918981e-01	 1.3469140e+00	 1.4886210e-01	[ 1.1464708e+00]	 2.1734834e-01


.. parsed-literal::

      72	 1.3048762e+00	 1.3920777e-01	 1.3554408e+00	 1.4847230e-01	  1.1437685e+00 	 2.0600986e-01


.. parsed-literal::

      73	 1.3097313e+00	 1.3854554e-01	 1.3606283e+00	 1.4759055e-01	  1.1278362e+00 	 2.1240640e-01


.. parsed-literal::

      74	 1.3168069e+00	 1.3875289e-01	 1.3675434e+00	 1.4726391e-01	  1.1327142e+00 	 2.1801615e-01


.. parsed-literal::

      75	 1.3205403e+00	 1.3885103e-01	 1.3712864e+00	 1.4704406e-01	  1.1307015e+00 	 2.0799708e-01


.. parsed-literal::

      76	 1.3248355e+00	 1.3909243e-01	 1.3756194e+00	 1.4643635e-01	  1.1270012e+00 	 2.0316863e-01


.. parsed-literal::

      77	 1.3300894e+00	 1.3878438e-01	 1.3809638e+00	 1.4575896e-01	  1.1188932e+00 	 2.0235395e-01
      78	 1.3351298e+00	 1.3854806e-01	 1.3860361e+00	 1.4526093e-01	  1.1221644e+00 	 1.9585729e-01


.. parsed-literal::

      79	 1.3416283e+00	 1.3777331e-01	 1.3926886e+00	 1.4452957e-01	  1.1229707e+00 	 2.0303202e-01
      80	 1.3472704e+00	 1.3693340e-01	 1.3985452e+00	 1.4444047e-01	  1.1326681e+00 	 1.9681454e-01


.. parsed-literal::

      81	 1.3537564e+00	 1.3628889e-01	 1.4051716e+00	 1.4412933e-01	  1.1288849e+00 	 2.0981741e-01


.. parsed-literal::

      82	 1.3619531e+00	 1.3529641e-01	 1.4134573e+00	 1.4353249e-01	  1.1287782e+00 	 2.1067190e-01


.. parsed-literal::

      83	 1.3666882e+00	 1.3505071e-01	 1.4182257e+00	 1.4315008e-01	  1.1239841e+00 	 2.1177816e-01


.. parsed-literal::

      84	 1.3711702e+00	 1.3469139e-01	 1.4226184e+00	 1.4268528e-01	  1.1280381e+00 	 2.0118403e-01
      85	 1.3759982e+00	 1.3434124e-01	 1.4274539e+00	 1.4233088e-01	  1.1327847e+00 	 2.0349789e-01


.. parsed-literal::

      86	 1.3805878e+00	 1.3362199e-01	 1.4322473e+00	 1.4150559e-01	  1.1248955e+00 	 1.9033599e-01


.. parsed-literal::

      87	 1.3860778e+00	 1.3332249e-01	 1.4379162e+00	 1.4122925e-01	  1.1277390e+00 	 2.5041246e-01
      88	 1.3925325e+00	 1.3278565e-01	 1.4446171e+00	 1.4097014e-01	  1.1259484e+00 	 1.9563007e-01


.. parsed-literal::

      89	 1.3939037e+00	 1.3286562e-01	 1.4463011e+00	 1.4014445e-01	  1.1281008e+00 	 2.0962501e-01
      90	 1.4003008e+00	 1.3233273e-01	 1.4524917e+00	 1.3999497e-01	  1.1303359e+00 	 1.7023492e-01


.. parsed-literal::

      91	 1.4032495e+00	 1.3204284e-01	 1.4553201e+00	 1.3956032e-01	  1.1317307e+00 	 2.0695996e-01


.. parsed-literal::

      92	 1.4072857e+00	 1.3173906e-01	 1.4592755e+00	 1.3863038e-01	  1.1371889e+00 	 2.0977116e-01


.. parsed-literal::

      93	 1.4129409e+00	 1.3150293e-01	 1.4648656e+00	 1.3740678e-01	[ 1.1510584e+00]	 2.0801425e-01


.. parsed-literal::

      94	 1.4146033e+00	 1.3116304e-01	 1.4668104e+00	 1.3686571e-01	[ 1.1643642e+00]	 2.0857239e-01


.. parsed-literal::

      95	 1.4233736e+00	 1.3113407e-01	 1.4753039e+00	 1.3644019e-01	[ 1.1774787e+00]	 2.1659160e-01


.. parsed-literal::

      96	 1.4263434e+00	 1.3110882e-01	 1.4782984e+00	 1.3656005e-01	[ 1.1823010e+00]	 2.1869826e-01
      97	 1.4299369e+00	 1.3110218e-01	 1.4819178e+00	 1.3679152e-01	[ 1.1902792e+00]	 1.8579078e-01


.. parsed-literal::

      98	 1.4356250e+00	 1.3120946e-01	 1.4875877e+00	 1.3707119e-01	[ 1.2027869e+00]	 2.1000648e-01


.. parsed-literal::

      99	 1.4395751e+00	 1.3128307e-01	 1.4915921e+00	 1.3700061e-01	[ 1.2078054e+00]	 2.9123044e-01
     100	 1.4447447e+00	 1.3140293e-01	 1.4967330e+00	 1.3732408e-01	[ 1.2143160e+00]	 1.9517493e-01


.. parsed-literal::

     101	 1.4479600e+00	 1.3113130e-01	 1.4999415e+00	 1.3717477e-01	  1.2125692e+00 	 2.0141816e-01


.. parsed-literal::

     102	 1.4517042e+00	 1.3072061e-01	 1.5038894e+00	 1.3675803e-01	  1.2061440e+00 	 2.0606089e-01


.. parsed-literal::

     103	 1.4548643e+00	 1.3024014e-01	 1.5072651e+00	 1.3596304e-01	  1.1976267e+00 	 2.0377040e-01


.. parsed-literal::

     104	 1.4581480e+00	 1.2991779e-01	 1.5105953e+00	 1.3566600e-01	  1.1945331e+00 	 2.1849942e-01


.. parsed-literal::

     105	 1.4611823e+00	 1.2970091e-01	 1.5137319e+00	 1.3556466e-01	  1.1914165e+00 	 2.1623755e-01
     106	 1.4638979e+00	 1.2946473e-01	 1.5165313e+00	 1.3542387e-01	  1.1859356e+00 	 1.7980838e-01


.. parsed-literal::

     107	 1.4672435e+00	 1.2920876e-01	 1.5200523e+00	 1.3555973e-01	  1.1670812e+00 	 2.0268869e-01
     108	 1.4706872e+00	 1.2899839e-01	 1.5234450e+00	 1.3544778e-01	  1.1670542e+00 	 1.8505096e-01


.. parsed-literal::

     109	 1.4724776e+00	 1.2888429e-01	 1.5252369e+00	 1.3544000e-01	  1.1652275e+00 	 1.9380331e-01


.. parsed-literal::

     110	 1.4754314e+00	 1.2864863e-01	 1.5282305e+00	 1.3537864e-01	  1.1629265e+00 	 2.0208311e-01


.. parsed-literal::

     111	 1.4771698e+00	 1.2830891e-01	 1.5301795e+00	 1.3534630e-01	  1.1556565e+00 	 2.0920181e-01


.. parsed-literal::

     112	 1.4814036e+00	 1.2818387e-01	 1.5342332e+00	 1.3502302e-01	  1.1633974e+00 	 2.1700382e-01
     113	 1.4833321e+00	 1.2805304e-01	 1.5361660e+00	 1.3482944e-01	  1.1615181e+00 	 1.9168925e-01


.. parsed-literal::

     114	 1.4854552e+00	 1.2793728e-01	 1.5382833e+00	 1.3460580e-01	  1.1607906e+00 	 2.0968509e-01


.. parsed-literal::

     115	 1.4886641e+00	 1.2770844e-01	 1.5415419e+00	 1.3442493e-01	  1.1536565e+00 	 2.1721506e-01
     116	 1.4919995e+00	 1.2764965e-01	 1.5448669e+00	 1.3437088e-01	  1.1507746e+00 	 1.9452429e-01


.. parsed-literal::

     117	 1.4941798e+00	 1.2751596e-01	 1.5470432e+00	 1.3434296e-01	  1.1471024e+00 	 2.0714855e-01


.. parsed-literal::

     118	 1.4968768e+00	 1.2728189e-01	 1.5498193e+00	 1.3443165e-01	  1.1400182e+00 	 2.1452618e-01


.. parsed-literal::

     119	 1.4987829e+00	 1.2708461e-01	 1.5518196e+00	 1.3416699e-01	  1.1315457e+00 	 2.0732307e-01


.. parsed-literal::

     120	 1.5009810e+00	 1.2693868e-01	 1.5540581e+00	 1.3408428e-01	  1.1276491e+00 	 2.0217085e-01
     121	 1.5042035e+00	 1.2668844e-01	 1.5573918e+00	 1.3390542e-01	  1.1162788e+00 	 2.0085311e-01


.. parsed-literal::

     122	 1.5057569e+00	 1.2657055e-01	 1.5589857e+00	 1.3405988e-01	  1.1127123e+00 	 2.0177031e-01
     123	 1.5072397e+00	 1.2657364e-01	 1.5603986e+00	 1.3409357e-01	  1.1144913e+00 	 1.9677877e-01


.. parsed-literal::

     124	 1.5099224e+00	 1.2653009e-01	 1.5630313e+00	 1.3430020e-01	  1.1124016e+00 	 1.9888854e-01


.. parsed-literal::

     125	 1.5116586e+00	 1.2653526e-01	 1.5648177e+00	 1.3446190e-01	  1.1085809e+00 	 2.0346904e-01


.. parsed-literal::

     126	 1.5128844e+00	 1.2644963e-01	 1.5663204e+00	 1.3459822e-01	  1.0899286e+00 	 2.0824552e-01


.. parsed-literal::

     127	 1.5162625e+00	 1.2647486e-01	 1.5697114e+00	 1.3469376e-01	  1.0944352e+00 	 2.0950341e-01


.. parsed-literal::

     128	 1.5172709e+00	 1.2639576e-01	 1.5707089e+00	 1.3448451e-01	  1.0971773e+00 	 2.1486521e-01


.. parsed-literal::

     129	 1.5192308e+00	 1.2624140e-01	 1.5727747e+00	 1.3431392e-01	  1.0946874e+00 	 2.0926976e-01
     130	 1.5210019e+00	 1.2602588e-01	 1.5746644e+00	 1.3389623e-01	  1.0921571e+00 	 1.9985700e-01


.. parsed-literal::

     131	 1.5233911e+00	 1.2592819e-01	 1.5770490e+00	 1.3397480e-01	  1.0906884e+00 	 2.0502877e-01


.. parsed-literal::

     132	 1.5251207e+00	 1.2584314e-01	 1.5787543e+00	 1.3400323e-01	  1.0886648e+00 	 2.0373535e-01


.. parsed-literal::

     133	 1.5269712e+00	 1.2572071e-01	 1.5805907e+00	 1.3386790e-01	  1.0871928e+00 	 2.1157455e-01


.. parsed-literal::

     134	 1.5288562e+00	 1.2540894e-01	 1.5825714e+00	 1.3340552e-01	  1.0801345e+00 	 2.1405292e-01
     135	 1.5313689e+00	 1.2531336e-01	 1.5850308e+00	 1.3327362e-01	  1.0831446e+00 	 1.9973493e-01


.. parsed-literal::

     136	 1.5329093e+00	 1.2522636e-01	 1.5865911e+00	 1.3321409e-01	  1.0830071e+00 	 2.0887589e-01


.. parsed-literal::

     137	 1.5346782e+00	 1.2507677e-01	 1.5884611e+00	 1.3317930e-01	  1.0802948e+00 	 2.1209121e-01
     138	 1.5359691e+00	 1.2486766e-01	 1.5898824e+00	 1.3339559e-01	  1.0706889e+00 	 1.8297839e-01


.. parsed-literal::

     139	 1.5376543e+00	 1.2483276e-01	 1.5915140e+00	 1.3344504e-01	  1.0739149e+00 	 2.1329570e-01
     140	 1.5390823e+00	 1.2479127e-01	 1.5928981e+00	 1.3362171e-01	  1.0754633e+00 	 1.7019677e-01


.. parsed-literal::

     141	 1.5402987e+00	 1.2472298e-01	 1.5940898e+00	 1.3373489e-01	  1.0762334e+00 	 1.7631888e-01
     142	 1.5419723e+00	 1.2453267e-01	 1.5958028e+00	 1.3370616e-01	  1.0713191e+00 	 1.9545770e-01


.. parsed-literal::

     143	 1.5442664e+00	 1.2446716e-01	 1.5980887e+00	 1.3379527e-01	  1.0756448e+00 	 2.1090794e-01
     144	 1.5453133e+00	 1.2440884e-01	 1.5991510e+00	 1.3363706e-01	  1.0752412e+00 	 1.9919634e-01


.. parsed-literal::

     145	 1.5466378e+00	 1.2433722e-01	 1.6005555e+00	 1.3346442e-01	  1.0739080e+00 	 2.1412754e-01


.. parsed-literal::

     146	 1.5484267e+00	 1.2422130e-01	 1.6024336e+00	 1.3318298e-01	  1.0727126e+00 	 2.0404577e-01


.. parsed-literal::

     147	 1.5502868e+00	 1.2422885e-01	 1.6044563e+00	 1.3326514e-01	  1.0745540e+00 	 2.1715975e-01
     148	 1.5518308e+00	 1.2413689e-01	 1.6059650e+00	 1.3315579e-01	  1.0769613e+00 	 1.7931223e-01


.. parsed-literal::

     149	 1.5531484e+00	 1.2404344e-01	 1.6072270e+00	 1.3317687e-01	  1.0777184e+00 	 1.9730663e-01


.. parsed-literal::

     150	 1.5546521e+00	 1.2393850e-01	 1.6087191e+00	 1.3330170e-01	  1.0757995e+00 	 2.1634912e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.03 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f16fc160be0>



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
    CPU times: user 1.7 s, sys: 56.9 ms, total: 1.76 s
    Wall time: 549 ms


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

