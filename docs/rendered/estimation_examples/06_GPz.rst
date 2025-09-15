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
       1	-3.4391817e-01	 3.2065578e-01	-3.3426273e-01	 3.2016225e-01	[-3.3361710e-01]	 4.5857692e-01


.. parsed-literal::

       2	-2.7295725e-01	 3.1003256e-01	-2.4899987e-01	 3.0941335e-01	[-2.4757198e-01]	 2.3117518e-01


.. parsed-literal::

       3	-2.2917234e-01	 2.8958030e-01	-1.8716666e-01	 2.8782109e-01	[-1.8128043e-01]	 2.8958392e-01
       4	-1.9327590e-01	 2.6632907e-01	-1.5255588e-01	 2.6147761e-01	[-1.2713531e-01]	 1.8028069e-01


.. parsed-literal::

       5	-1.0496497e-01	 2.5744381e-01	-7.1245547e-02	 2.5304079e-01	[-5.1488607e-02]	 2.0547056e-01
       6	-7.2124108e-02	 2.5230471e-01	-4.2447484e-02	 2.4616491e-01	[-2.1066152e-02]	 1.7886877e-01


.. parsed-literal::

       7	-5.4819299e-02	 2.4959550e-01	-3.1240301e-02	 2.4396798e-01	[-1.0038105e-02]	 1.9937062e-01


.. parsed-literal::

       8	-4.2770331e-02	 2.4758564e-01	-2.2862252e-02	 2.4262979e-01	[-3.6946626e-03]	 2.1404672e-01
       9	-3.0022189e-02	 2.4521971e-01	-1.2765209e-02	 2.4098476e-01	[ 3.8150209e-03]	 2.0012927e-01


.. parsed-literal::

      10	-1.8615002e-02	 2.4304396e-01	-3.2571839e-03	 2.3950728e-01	[ 1.0788967e-02]	 2.5501990e-01


.. parsed-literal::

      11	-1.4911767e-02	 2.4260343e-01	-6.7662623e-04	 2.3900009e-01	[ 1.2998413e-02]	 2.1013236e-01


.. parsed-literal::

      12	-1.1378283e-02	 2.4184263e-01	 2.5992044e-03	 2.3831010e-01	[ 1.6638600e-02]	 2.0808983e-01


.. parsed-literal::

      13	-8.0765784e-03	 2.4113444e-01	 5.8677063e-03	 2.3778516e-01	[ 1.8876450e-02]	 2.1265125e-01


.. parsed-literal::

      14	-3.8616142e-03	 2.4021486e-01	 1.0654960e-02	 2.3673096e-01	[ 2.4718252e-02]	 2.0639038e-01


.. parsed-literal::

      15	 1.2521635e-01	 2.2702500e-01	 1.4823826e-01	 2.2396774e-01	[ 1.5538651e-01]	 3.1769633e-01


.. parsed-literal::

      16	 1.9568855e-01	 2.2375186e-01	 2.1927433e-01	 2.2046083e-01	[ 2.2616718e-01]	 3.2762933e-01


.. parsed-literal::

      17	 2.3975426e-01	 2.2002812e-01	 2.6474240e-01	 2.1676142e-01	[ 2.7228231e-01]	 2.1654177e-01
      18	 3.0176263e-01	 2.1621728e-01	 3.3108331e-01	 2.1430477e-01	[ 3.2967683e-01]	 1.8178487e-01


.. parsed-literal::

      19	 3.3445962e-01	 2.1507353e-01	 3.6509968e-01	 2.1278512e-01	[ 3.6788905e-01]	 2.0937824e-01
      20	 3.7484419e-01	 2.1346068e-01	 4.0690055e-01	 2.1071850e-01	[ 4.0948641e-01]	 1.9732046e-01


.. parsed-literal::

      21	 4.5154380e-01	 2.1519146e-01	 4.8423696e-01	 2.1282081e-01	[ 4.8044801e-01]	 2.0409679e-01


.. parsed-literal::

      22	 5.3039435e-01	 2.1569535e-01	 5.6548655e-01	 2.1321195e-01	[ 5.6085306e-01]	 2.2064042e-01


.. parsed-literal::

      23	 5.7087164e-01	 2.1299459e-01	 6.0967382e-01	 2.1013438e-01	[ 5.9211083e-01]	 2.1019888e-01


.. parsed-literal::

      24	 6.1369293e-01	 2.0996749e-01	 6.5108179e-01	 2.0808580e-01	[ 6.3207285e-01]	 2.1106553e-01


.. parsed-literal::

      25	 6.4434752e-01	 2.0916364e-01	 6.8116314e-01	 2.0658795e-01	[ 6.5910771e-01]	 2.0462275e-01


.. parsed-literal::

      26	 6.8620548e-01	 2.0979084e-01	 7.2248931e-01	 2.0594598e-01	[ 6.9426387e-01]	 2.1952534e-01


.. parsed-literal::

      27	 7.0497513e-01	 2.0982217e-01	 7.4116451e-01	 2.0689321e-01	[ 7.0433531e-01]	 2.1124554e-01


.. parsed-literal::

      28	 7.4336572e-01	 2.0929573e-01	 7.8165241e-01	 2.0525598e-01	[ 7.3079404e-01]	 2.1794605e-01
      29	 7.6017418e-01	 2.0842440e-01	 7.9790374e-01	 2.0442589e-01	[ 7.4835118e-01]	 1.8987656e-01


.. parsed-literal::

      30	 7.8309848e-01	 2.0807661e-01	 8.2126766e-01	 2.0391109e-01	[ 7.6281363e-01]	 2.1275520e-01


.. parsed-literal::

      31	 8.1124172e-01	 2.0810629e-01	 8.5017416e-01	 2.0211861e-01	[ 7.8862378e-01]	 2.1940613e-01
      32	 8.4014634e-01	 2.0804229e-01	 8.8099363e-01	 2.0135280e-01	[ 8.1643719e-01]	 1.9503999e-01


.. parsed-literal::

      33	 8.5200844e-01	 2.1518430e-01	 8.9381062e-01	 2.0872611e-01	[ 8.4611425e-01]	 2.0513797e-01


.. parsed-literal::

      34	 8.7974631e-01	 2.0922388e-01	 9.2172313e-01	 2.0336143e-01	[ 8.7012875e-01]	 2.1471381e-01


.. parsed-literal::

      35	 8.9349468e-01	 2.0900095e-01	 9.3503261e-01	 2.0351853e-01	[ 8.8283831e-01]	 2.0975566e-01


.. parsed-literal::

      36	 9.0860562e-01	 2.0997596e-01	 9.5032749e-01	 2.0465424e-01	[ 8.9706792e-01]	 2.0998383e-01
      37	 9.2204168e-01	 2.1338170e-01	 9.6465555e-01	 2.0886669e-01	[ 9.1759041e-01]	 1.9863868e-01


.. parsed-literal::

      38	 9.3361949e-01	 2.1271521e-01	 9.7619758e-01	 2.0847595e-01	[ 9.2940610e-01]	 1.8066216e-01


.. parsed-literal::

      39	 9.4462158e-01	 2.1158159e-01	 9.8758659e-01	 2.0800780e-01	[ 9.3931683e-01]	 2.0633149e-01


.. parsed-literal::

      40	 9.5933941e-01	 2.0947795e-01	 1.0026552e+00	 2.0671027e-01	[ 9.4836014e-01]	 2.0567918e-01
      41	 9.8002410e-01	 2.0714312e-01	 1.0249657e+00	 2.0460571e-01	[ 9.5990637e-01]	 1.9295216e-01


.. parsed-literal::

      42	 9.9103506e-01	 2.0422246e-01	 1.0366531e+00	 2.0241771e-01	[ 9.6082800e-01]	 1.9051266e-01


.. parsed-literal::

      43	 9.9934618e-01	 2.0425064e-01	 1.0448058e+00	 2.0193180e-01	[ 9.7047882e-01]	 2.0871019e-01
      44	 1.0097800e+00	 2.0363827e-01	 1.0556234e+00	 2.0092639e-01	[ 9.7836322e-01]	 1.9908214e-01


.. parsed-literal::

      45	 1.0226895e+00	 2.0200168e-01	 1.0692163e+00	 1.9891315e-01	[ 9.8315266e-01]	 2.1618843e-01


.. parsed-literal::

      46	 1.0310193e+00	 2.0196717e-01	 1.0779277e+00	 1.9867061e-01	[ 9.8543740e-01]	 2.1525836e-01
      47	 1.0418378e+00	 1.9941958e-01	 1.0885143e+00	 1.9557974e-01	[ 9.9770911e-01]	 1.8418622e-01


.. parsed-literal::

      48	 1.0493928e+00	 1.9805667e-01	 1.0961523e+00	 1.9376898e-01	[ 1.0058053e+00]	 2.1295881e-01
      49	 1.0585952e+00	 1.9673434e-01	 1.1053381e+00	 1.9196901e-01	[ 1.0172154e+00]	 1.7759609e-01


.. parsed-literal::

      50	 1.0714081e+00	 1.9503718e-01	 1.1184355e+00	 1.8940432e-01	[ 1.0315962e+00]	 1.9544411e-01


.. parsed-literal::

      51	 1.0764749e+00	 1.9497226e-01	 1.1236793e+00	 1.8865537e-01	[ 1.0387879e+00]	 2.0119333e-01


.. parsed-literal::

      52	 1.0861203e+00	 1.9402864e-01	 1.1329676e+00	 1.8786903e-01	[ 1.0472387e+00]	 2.0651841e-01
      53	 1.0914978e+00	 1.9341517e-01	 1.1384730e+00	 1.8692977e-01	[ 1.0509365e+00]	 1.7606258e-01


.. parsed-literal::

      54	 1.0998434e+00	 1.9291702e-01	 1.1471215e+00	 1.8623194e-01	[ 1.0580045e+00]	 2.0881724e-01


.. parsed-literal::

      55	 1.1074844e+00	 1.9300088e-01	 1.1550159e+00	 1.8538563e-01	[ 1.0679498e+00]	 2.0366359e-01


.. parsed-literal::

      56	 1.1144962e+00	 1.9357260e-01	 1.1623426e+00	 1.8644659e-01	[ 1.0754283e+00]	 2.1509647e-01
      57	 1.1187091e+00	 1.9317662e-01	 1.1665657e+00	 1.8603523e-01	[ 1.0791916e+00]	 2.0074439e-01


.. parsed-literal::

      58	 1.1307899e+00	 1.9204978e-01	 1.1790471e+00	 1.8521408e-01	[ 1.0883524e+00]	 2.1076608e-01


.. parsed-literal::

      59	 1.1365509e+00	 1.9122072e-01	 1.1849418e+00	 1.8448434e-01	[ 1.0985213e+00]	 2.0607257e-01


.. parsed-literal::

      60	 1.1451093e+00	 1.9111253e-01	 1.1935413e+00	 1.8456201e-01	[ 1.1103164e+00]	 2.0355535e-01


.. parsed-literal::

      61	 1.1523543e+00	 1.9039766e-01	 1.2009818e+00	 1.8406399e-01	[ 1.1170858e+00]	 2.1763730e-01


.. parsed-literal::

      62	 1.1596335e+00	 1.8936614e-01	 1.2087054e+00	 1.8309083e-01	[ 1.1202460e+00]	 2.0676637e-01
      63	 1.1695530e+00	 1.8696578e-01	 1.2187841e+00	 1.8072422e-01	[ 1.1268952e+00]	 1.9331598e-01


.. parsed-literal::

      64	 1.1767482e+00	 1.8497774e-01	 1.2263920e+00	 1.7857449e-01	[ 1.1284900e+00]	 2.1303296e-01


.. parsed-literal::

      65	 1.1848679e+00	 1.8371706e-01	 1.2341986e+00	 1.7718612e-01	[ 1.1367496e+00]	 2.1872473e-01
      66	 1.1913898e+00	 1.8178278e-01	 1.2408441e+00	 1.7511902e-01	[ 1.1412944e+00]	 1.9983101e-01


.. parsed-literal::

      67	 1.1978480e+00	 1.8012005e-01	 1.2473249e+00	 1.7328218e-01	[ 1.1457707e+00]	 2.1715856e-01
      68	 1.2060402e+00	 1.7708967e-01	 1.2558505e+00	 1.7019448e-01	[ 1.1483745e+00]	 1.9309902e-01


.. parsed-literal::

      69	 1.2142191e+00	 1.7562063e-01	 1.2638665e+00	 1.6880446e-01	[ 1.1538345e+00]	 1.7415309e-01


.. parsed-literal::

      70	 1.2210952e+00	 1.7445387e-01	 1.2707484e+00	 1.6787069e-01	[ 1.1596619e+00]	 2.1473408e-01


.. parsed-literal::

      71	 1.2316133e+00	 1.7282814e-01	 1.2814627e+00	 1.6666991e-01	[ 1.1696238e+00]	 2.0209265e-01


.. parsed-literal::

      72	 1.2369468e+00	 1.7164066e-01	 1.2875036e+00	 1.6595076e-01	[ 1.1790414e+00]	 2.0593739e-01


.. parsed-literal::

      73	 1.2459905e+00	 1.7182967e-01	 1.2962010e+00	 1.6618919e-01	[ 1.1947187e+00]	 2.0773888e-01


.. parsed-literal::

      74	 1.2504374e+00	 1.7106765e-01	 1.3005922e+00	 1.6530839e-01	[ 1.2021841e+00]	 2.0830345e-01


.. parsed-literal::

      75	 1.2558945e+00	 1.7045523e-01	 1.3062050e+00	 1.6466694e-01	[ 1.2114893e+00]	 2.1177554e-01
      76	 1.2629806e+00	 1.6925766e-01	 1.3138736e+00	 1.6380844e-01	[ 1.2249652e+00]	 1.7534733e-01


.. parsed-literal::

      77	 1.2677583e+00	 1.6774876e-01	 1.3189464e+00	 1.6242083e-01	[ 1.2306836e+00]	 1.9826078e-01
      78	 1.2723330e+00	 1.6734428e-01	 1.3233623e+00	 1.6207017e-01	[ 1.2360882e+00]	 1.8701172e-01


.. parsed-literal::

      79	 1.2785651e+00	 1.6628308e-01	 1.3296649e+00	 1.6120224e-01	[ 1.2438076e+00]	 1.9917035e-01
      80	 1.2847698e+00	 1.6505831e-01	 1.3360281e+00	 1.6033199e-01	[ 1.2506139e+00]	 1.9101882e-01


.. parsed-literal::

      81	 1.2878661e+00	 1.6290804e-01	 1.3398113e+00	 1.5892881e-01	[ 1.2512048e+00]	 2.0747805e-01
      82	 1.2962226e+00	 1.6261582e-01	 1.3479086e+00	 1.5862380e-01	[ 1.2573283e+00]	 1.8481445e-01


.. parsed-literal::

      83	 1.2996303e+00	 1.6220024e-01	 1.3513349e+00	 1.5828901e-01	[ 1.2582343e+00]	 2.1559739e-01
      84	 1.3045232e+00	 1.6136159e-01	 1.3564071e+00	 1.5769551e-01	  1.2578675e+00 	 1.7505407e-01


.. parsed-literal::

      85	 1.3089332e+00	 1.6002129e-01	 1.3609973e+00	 1.5668172e-01	  1.2582208e+00 	 2.1582818e-01


.. parsed-literal::

      86	 1.3138232e+00	 1.5959707e-01	 1.3659762e+00	 1.5642005e-01	[ 1.2611267e+00]	 2.1882725e-01


.. parsed-literal::

      87	 1.3185169e+00	 1.5944432e-01	 1.3706443e+00	 1.5624602e-01	[ 1.2665335e+00]	 2.0640564e-01


.. parsed-literal::

      88	 1.3233528e+00	 1.5883128e-01	 1.3755596e+00	 1.5548289e-01	[ 1.2712430e+00]	 2.0663357e-01


.. parsed-literal::

      89	 1.3263066e+00	 1.5893195e-01	 1.3787759e+00	 1.5550082e-01	[ 1.2752024e+00]	 2.0806670e-01


.. parsed-literal::

      90	 1.3304546e+00	 1.5803208e-01	 1.3828686e+00	 1.5450556e-01	[ 1.2775914e+00]	 2.1068454e-01


.. parsed-literal::

      91	 1.3335066e+00	 1.5741346e-01	 1.3859043e+00	 1.5385291e-01	[ 1.2796042e+00]	 2.0200515e-01


.. parsed-literal::

      92	 1.3374216e+00	 1.5660938e-01	 1.3898891e+00	 1.5304711e-01	[ 1.2819629e+00]	 2.1834087e-01
      93	 1.3430241e+00	 1.5538395e-01	 1.3955726e+00	 1.5175763e-01	[ 1.2864078e+00]	 1.8325591e-01


.. parsed-literal::

      94	 1.3466333e+00	 1.5470234e-01	 1.3992594e+00	 1.5114708e-01	[ 1.2896095e+00]	 3.1162739e-01


.. parsed-literal::

      95	 1.3510205e+00	 1.5395164e-01	 1.4036628e+00	 1.5035009e-01	[ 1.2948726e+00]	 2.1796393e-01
      96	 1.3546570e+00	 1.5339436e-01	 1.4073777e+00	 1.4982789e-01	[ 1.2984611e+00]	 1.8854713e-01


.. parsed-literal::

      97	 1.3587341e+00	 1.5279685e-01	 1.4115651e+00	 1.4942425e-01	[ 1.3001842e+00]	 2.0588923e-01
      98	 1.3612434e+00	 1.5209205e-01	 1.4144499e+00	 1.4913599e-01	  1.2963082e+00 	 1.7950535e-01


.. parsed-literal::

      99	 1.3658876e+00	 1.5173288e-01	 1.4189766e+00	 1.4895914e-01	[ 1.3011336e+00]	 2.0264006e-01
     100	 1.3679953e+00	 1.5154621e-01	 1.4210437e+00	 1.4881993e-01	[ 1.3021777e+00]	 1.9777441e-01


.. parsed-literal::

     101	 1.3704706e+00	 1.5118356e-01	 1.4236150e+00	 1.4857714e-01	[ 1.3041044e+00]	 2.1623707e-01


.. parsed-literal::

     102	 1.3716320e+00	 1.5076952e-01	 1.4249900e+00	 1.4802917e-01	  1.2973851e+00 	 2.0695782e-01


.. parsed-literal::

     103	 1.3749964e+00	 1.5072452e-01	 1.4282739e+00	 1.4803100e-01	[ 1.3053320e+00]	 2.1486831e-01


.. parsed-literal::

     104	 1.3770238e+00	 1.5054834e-01	 1.4303490e+00	 1.4782703e-01	[ 1.3088900e+00]	 2.2090936e-01


.. parsed-literal::

     105	 1.3789461e+00	 1.5034701e-01	 1.4323154e+00	 1.4753417e-01	[ 1.3106186e+00]	 2.0449257e-01


.. parsed-literal::

     106	 1.3824363e+00	 1.4996314e-01	 1.4358708e+00	 1.4717835e-01	  1.3096640e+00 	 2.0088577e-01


.. parsed-literal::

     107	 1.3851621e+00	 1.4929935e-01	 1.4386843e+00	 1.4655069e-01	  1.3030582e+00 	 3.2391453e-01


.. parsed-literal::

     108	 1.3895986e+00	 1.4891359e-01	 1.4431458e+00	 1.4644113e-01	  1.2973158e+00 	 2.0816422e-01


.. parsed-literal::

     109	 1.3927529e+00	 1.4862204e-01	 1.4462681e+00	 1.4637296e-01	  1.2945007e+00 	 2.1093178e-01


.. parsed-literal::

     110	 1.3964327e+00	 1.4827132e-01	 1.4499696e+00	 1.4632878e-01	  1.2932278e+00 	 2.0546937e-01


.. parsed-literal::

     111	 1.3976065e+00	 1.4790621e-01	 1.4513902e+00	 1.4618860e-01	  1.2830335e+00 	 2.0645046e-01


.. parsed-literal::

     112	 1.4011046e+00	 1.4786320e-01	 1.4547473e+00	 1.4611058e-01	  1.2923226e+00 	 2.0622849e-01
     113	 1.4031410e+00	 1.4777525e-01	 1.4567838e+00	 1.4600401e-01	  1.2960836e+00 	 1.9905090e-01


.. parsed-literal::

     114	 1.4056899e+00	 1.4760401e-01	 1.4593935e+00	 1.4585419e-01	  1.2984847e+00 	 2.0523024e-01


.. parsed-literal::

     115	 1.4092426e+00	 1.4726898e-01	 1.4630360e+00	 1.4566733e-01	  1.2953709e+00 	 2.0402765e-01


.. parsed-literal::

     116	 1.4113419e+00	 1.4721074e-01	 1.4652324e+00	 1.4560719e-01	  1.2979953e+00 	 2.9346848e-01


.. parsed-literal::

     117	 1.4137646e+00	 1.4699409e-01	 1.4676970e+00	 1.4551575e-01	  1.2928315e+00 	 2.1703911e-01


.. parsed-literal::

     118	 1.4156756e+00	 1.4681476e-01	 1.4696335e+00	 1.4530788e-01	  1.2879948e+00 	 2.1006513e-01


.. parsed-literal::

     119	 1.4177413e+00	 1.4688380e-01	 1.4717957e+00	 1.4534364e-01	  1.2916872e+00 	 2.1059322e-01


.. parsed-literal::

     120	 1.4197355e+00	 1.4668894e-01	 1.4738114e+00	 1.4494490e-01	  1.2910977e+00 	 2.1055794e-01


.. parsed-literal::

     121	 1.4214474e+00	 1.4663158e-01	 1.4755307e+00	 1.4472997e-01	  1.2948662e+00 	 2.1955037e-01


.. parsed-literal::

     122	 1.4236092e+00	 1.4648173e-01	 1.4777247e+00	 1.4448093e-01	  1.2983798e+00 	 2.1262121e-01
     123	 1.4250271e+00	 1.4626574e-01	 1.4792220e+00	 1.4423207e-01	  1.3010561e+00 	 1.8295431e-01


.. parsed-literal::

     124	 1.4276063e+00	 1.4603549e-01	 1.4817692e+00	 1.4412967e-01	  1.3001643e+00 	 1.8765140e-01
     125	 1.4290763e+00	 1.4580508e-01	 1.4832161e+00	 1.4405085e-01	  1.2980345e+00 	 1.9467759e-01


.. parsed-literal::

     126	 1.4309583e+00	 1.4546454e-01	 1.4851143e+00	 1.4388530e-01	  1.2946687e+00 	 2.0501494e-01


.. parsed-literal::

     127	 1.4339004e+00	 1.4501114e-01	 1.4881552e+00	 1.4365548e-01	  1.2881211e+00 	 2.1323109e-01
     128	 1.4364252e+00	 1.4435420e-01	 1.4907482e+00	 1.4325731e-01	  1.2837744e+00 	 1.7195845e-01


.. parsed-literal::

     129	 1.4381179e+00	 1.4439532e-01	 1.4923893e+00	 1.4323062e-01	  1.2892199e+00 	 2.0622516e-01


.. parsed-literal::

     130	 1.4394545e+00	 1.4432170e-01	 1.4937170e+00	 1.4306883e-01	  1.2914798e+00 	 2.1019435e-01


.. parsed-literal::

     131	 1.4414991e+00	 1.4409834e-01	 1.4957566e+00	 1.4286171e-01	  1.2964733e+00 	 2.0942974e-01


.. parsed-literal::

     132	 1.4427488e+00	 1.4340612e-01	 1.4970847e+00	 1.4200265e-01	  1.2862097e+00 	 2.1340203e-01
     133	 1.4455627e+00	 1.4337150e-01	 1.4997799e+00	 1.4216260e-01	  1.2959917e+00 	 1.7877603e-01


.. parsed-literal::

     134	 1.4465135e+00	 1.4318598e-01	 1.5006978e+00	 1.4207356e-01	  1.2969780e+00 	 2.0775771e-01


.. parsed-literal::

     135	 1.4478672e+00	 1.4285413e-01	 1.5020441e+00	 1.4184943e-01	  1.2970643e+00 	 2.1934223e-01


.. parsed-literal::

     136	 1.4491794e+00	 1.4218300e-01	 1.5033831e+00	 1.4126965e-01	  1.2978108e+00 	 2.0936489e-01


.. parsed-literal::

     137	 1.4507669e+00	 1.4203217e-01	 1.5049643e+00	 1.4113322e-01	  1.2985894e+00 	 2.0993638e-01


.. parsed-literal::

     138	 1.4525708e+00	 1.4180317e-01	 1.5068355e+00	 1.4088932e-01	  1.2986544e+00 	 2.0234966e-01
     139	 1.4540484e+00	 1.4160162e-01	 1.5083767e+00	 1.4074139e-01	  1.2982611e+00 	 1.9683242e-01


.. parsed-literal::

     140	 1.4555186e+00	 1.4152329e-01	 1.5099299e+00	 1.4071113e-01	  1.2963246e+00 	 3.1207061e-01
     141	 1.4576863e+00	 1.4124428e-01	 1.5121548e+00	 1.4063907e-01	  1.2930331e+00 	 1.8716145e-01


.. parsed-literal::

     142	 1.4588671e+00	 1.4119375e-01	 1.5133150e+00	 1.4065723e-01	  1.2931444e+00 	 2.0755482e-01


.. parsed-literal::

     143	 1.4608136e+00	 1.4118639e-01	 1.5152798e+00	 1.4079192e-01	  1.2884211e+00 	 2.1470547e-01
     144	 1.4626273e+00	 1.4109061e-01	 1.5171604e+00	 1.4074773e-01	  1.2880251e+00 	 2.0218420e-01


.. parsed-literal::

     145	 1.4645857e+00	 1.4103716e-01	 1.5191712e+00	 1.4074615e-01	  1.2857070e+00 	 2.0800352e-01


.. parsed-literal::

     146	 1.4668835e+00	 1.4086189e-01	 1.5216031e+00	 1.4059658e-01	  1.2820900e+00 	 2.1846414e-01


.. parsed-literal::

     147	 1.4680042e+00	 1.4070850e-01	 1.5228922e+00	 1.4062400e-01	  1.2890427e+00 	 2.0200372e-01


.. parsed-literal::

     148	 1.4695455e+00	 1.4059364e-01	 1.5243656e+00	 1.4046927e-01	  1.2903120e+00 	 2.0249271e-01
     149	 1.4709542e+00	 1.4047204e-01	 1.5257793e+00	 1.4042487e-01	  1.2908716e+00 	 1.6909671e-01


.. parsed-literal::

     150	 1.4722526e+00	 1.4030711e-01	 1.5270893e+00	 1.4036063e-01	  1.2916484e+00 	 2.1173620e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.11 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5e444d9780>



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
    CPU times: user 2.04 s, sys: 44.9 ms, total: 2.09 s
    Wall time: 621 ms


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

