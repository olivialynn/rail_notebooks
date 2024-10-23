Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f02ae4072b0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>27.925796</td>
          <td>1.012862</td>
          <td>26.411825</td>
          <td>0.128337</td>
          <td>25.989770</td>
          <td>0.078138</td>
          <td>25.335720</td>
          <td>0.071495</td>
          <td>25.231618</td>
          <td>0.124144</td>
          <td>24.977735</td>
          <td>0.218823</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.240218</td>
          <td>0.650352</td>
          <td>28.901343</td>
          <td>0.875187</td>
          <td>27.628724</td>
          <td>0.314801</td>
          <td>27.623722</td>
          <td>0.480948</td>
          <td>26.448446</td>
          <td>0.342503</td>
          <td>28.142195</td>
          <td>1.771761</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.620180</td>
          <td>0.413625</td>
          <td>25.999991</td>
          <td>0.089602</td>
          <td>24.779915</td>
          <td>0.026804</td>
          <td>23.869425</td>
          <td>0.019714</td>
          <td>23.149005</td>
          <td>0.019934</td>
          <td>22.852193</td>
          <td>0.034222</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.241761</td>
          <td>1.215155</td>
          <td>27.646834</td>
          <td>0.357889</td>
          <td>27.647748</td>
          <td>0.319618</td>
          <td>26.754726</td>
          <td>0.242678</td>
          <td>26.014828</td>
          <td>0.241280</td>
          <td>24.810099</td>
          <td>0.190132</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.975635</td>
          <td>0.247836</td>
          <td>25.894284</td>
          <td>0.081646</td>
          <td>25.447999</td>
          <td>0.048338</td>
          <td>24.755870</td>
          <td>0.042742</td>
          <td>24.362835</td>
          <td>0.057805</td>
          <td>23.762217</td>
          <td>0.076653</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.686090</td>
          <td>0.434910</td>
          <td>26.170903</td>
          <td>0.104076</td>
          <td>26.252087</td>
          <td>0.098428</td>
          <td>26.065216</td>
          <td>0.135446</td>
          <td>26.189327</td>
          <td>0.278322</td>
          <td>25.542495</td>
          <td>0.346114</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.118875</td>
          <td>0.597386</td>
          <td>27.458091</td>
          <td>0.308164</td>
          <td>27.019621</td>
          <td>0.190738</td>
          <td>26.536918</td>
          <td>0.202457</td>
          <td>26.129006</td>
          <td>0.264984</td>
          <td>26.044721</td>
          <td>0.507671</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.936518</td>
          <td>1.019376</td>
          <td>27.021709</td>
          <td>0.215640</td>
          <td>27.143730</td>
          <td>0.211685</td>
          <td>26.661871</td>
          <td>0.224724</td>
          <td>26.633911</td>
          <td>0.395851</td>
          <td>25.289533</td>
          <td>0.282747</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.951544</td>
          <td>1.739752</td>
          <td>27.276467</td>
          <td>0.266079</td>
          <td>26.611437</td>
          <td>0.134589</td>
          <td>25.856370</td>
          <td>0.112994</td>
          <td>25.362112</td>
          <td>0.138979</td>
          <td>25.359124</td>
          <td>0.299087</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.287646</td>
          <td>1.246259</td>
          <td>26.447140</td>
          <td>0.132317</td>
          <td>26.270866</td>
          <td>0.100062</td>
          <td>25.558143</td>
          <td>0.087007</td>
          <td>25.116059</td>
          <td>0.112272</td>
          <td>25.073966</td>
          <td>0.237013</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>27.141205</td>
          <td>0.665998</td>
          <td>26.842210</td>
          <td>0.212638</td>
          <td>26.119429</td>
          <td>0.102958</td>
          <td>25.342727</td>
          <td>0.085225</td>
          <td>25.012808</td>
          <td>0.120426</td>
          <td>24.966369</td>
          <td>0.253642</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.081851</td>
          <td>0.561245</td>
          <td>27.405945</td>
          <td>0.305176</td>
          <td>26.995118</td>
          <td>0.344017</td>
          <td>27.100305</td>
          <td>0.638451</td>
          <td>25.678769</td>
          <td>0.445303</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.787438</td>
          <td>1.730704</td>
          <td>25.969215</td>
          <td>0.102657</td>
          <td>24.778092</td>
          <td>0.032183</td>
          <td>23.902525</td>
          <td>0.024481</td>
          <td>23.103294</td>
          <td>0.022956</td>
          <td>22.802843</td>
          <td>0.039694</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.323070</td>
          <td>0.332892</td>
          <td>28.206275</td>
          <td>0.593372</td>
          <td>26.447355</td>
          <td>0.235406</td>
          <td>25.954549</td>
          <td>0.284053</td>
          <td>25.044435</td>
          <td>0.288103</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.281331</td>
          <td>0.352871</td>
          <td>25.926744</td>
          <td>0.096972</td>
          <td>25.438806</td>
          <td>0.056490</td>
          <td>24.785064</td>
          <td>0.052045</td>
          <td>24.426393</td>
          <td>0.072010</td>
          <td>23.689169</td>
          <td>0.085034</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>26.486344</td>
          <td>0.419336</td>
          <td>26.314835</td>
          <td>0.138407</td>
          <td>26.170726</td>
          <td>0.109951</td>
          <td>25.884921</td>
          <td>0.139741</td>
          <td>26.180158</td>
          <td>0.326386</td>
          <td>24.903347</td>
          <td>0.245746</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.221138</td>
          <td>0.705066</td>
          <td>26.740234</td>
          <td>0.195928</td>
          <td>26.843944</td>
          <td>0.192825</td>
          <td>25.927676</td>
          <td>0.142529</td>
          <td>25.832556</td>
          <td>0.242495</td>
          <td>25.604730</td>
          <td>0.422457</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.156486</td>
          <td>1.255360</td>
          <td>27.215123</td>
          <td>0.291955</td>
          <td>26.635374</td>
          <td>0.162914</td>
          <td>26.188712</td>
          <td>0.179687</td>
          <td>26.779146</td>
          <td>0.512631</td>
          <td>25.509154</td>
          <td>0.395639</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.529308</td>
          <td>1.536243</td>
          <td>26.988800</td>
          <td>0.246624</td>
          <td>26.363982</td>
          <td>0.131399</td>
          <td>25.697619</td>
          <td>0.120097</td>
          <td>25.785203</td>
          <td>0.239241</td>
          <td>25.637626</td>
          <td>0.443728</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.654182</td>
          <td>0.472519</td>
          <td>26.405446</td>
          <td>0.148174</td>
          <td>26.254153</td>
          <td>0.116962</td>
          <td>25.691357</td>
          <td>0.116859</td>
          <td>25.158324</td>
          <td>0.137950</td>
          <td>25.320390</td>
          <td>0.340566</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>26.993375</td>
          <td>0.546126</td>
          <td>26.575123</td>
          <td>0.147752</td>
          <td>26.134721</td>
          <td>0.088799</td>
          <td>25.460287</td>
          <td>0.079827</td>
          <td>24.997354</td>
          <td>0.101223</td>
          <td>24.849523</td>
          <td>0.196577</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.375321</td>
          <td>0.713631</td>
          <td>27.985474</td>
          <td>0.464344</td>
          <td>28.030429</td>
          <td>0.431006</td>
          <td>27.033925</td>
          <td>0.304861</td>
          <td>25.898480</td>
          <td>0.219292</td>
          <td>25.638775</td>
          <td>0.373561</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.556823</td>
          <td>0.414657</td>
          <td>25.848808</td>
          <td>0.084321</td>
          <td>24.797039</td>
          <td>0.029533</td>
          <td>23.852531</td>
          <td>0.021124</td>
          <td>23.161911</td>
          <td>0.021832</td>
          <td>22.865081</td>
          <td>0.037709</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.669355</td>
          <td>0.434242</td>
          <td>27.846956</td>
          <td>0.454932</td>
          <td>26.726892</td>
          <td>0.294765</td>
          <td>25.895212</td>
          <td>0.269799</td>
          <td>25.384093</td>
          <td>0.375974</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.962709</td>
          <td>0.245444</td>
          <td>25.856417</td>
          <td>0.079065</td>
          <td>25.480746</td>
          <td>0.049836</td>
          <td>24.783083</td>
          <td>0.043853</td>
          <td>24.389988</td>
          <td>0.059299</td>
          <td>23.685000</td>
          <td>0.071702</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>2.147172</td>
          <td>25.861306</td>
          <td>0.237713</td>
          <td>26.118950</td>
          <td>0.106528</td>
          <td>26.242666</td>
          <td>0.105591</td>
          <td>26.483730</td>
          <td>0.209448</td>
          <td>25.730658</td>
          <td>0.205213</td>
          <td>25.236448</td>
          <td>0.291972</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.015782</td>
          <td>0.560228</td>
          <td>26.985235</td>
          <td>0.212041</td>
          <td>26.554519</td>
          <td>0.130219</td>
          <td>26.385540</td>
          <td>0.181180</td>
          <td>26.542076</td>
          <td>0.374041</td>
          <td>25.600265</td>
          <td>0.367732</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.844142</td>
          <td>0.503958</td>
          <td>27.941694</td>
          <td>0.465691</td>
          <td>27.534586</td>
          <td>0.305246</td>
          <td>26.227495</td>
          <td>0.163692</td>
          <td>26.029136</td>
          <td>0.255556</td>
          <td>25.431683</td>
          <td>0.331885</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.596123</td>
          <td>0.875851</td>
          <td>26.924368</td>
          <td>0.218805</td>
          <td>26.457557</td>
          <td>0.131900</td>
          <td>25.899457</td>
          <td>0.132048</td>
          <td>25.409014</td>
          <td>0.161817</td>
          <td>25.357522</td>
          <td>0.332706</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.105367</td>
          <td>0.604824</td>
          <td>26.481386</td>
          <td>0.140900</td>
          <td>26.017644</td>
          <td>0.083278</td>
          <td>25.668369</td>
          <td>0.099845</td>
          <td>25.157972</td>
          <td>0.121016</td>
          <td>24.732018</td>
          <td>0.185044</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
