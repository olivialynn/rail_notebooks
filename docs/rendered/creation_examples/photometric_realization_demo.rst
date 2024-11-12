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

    <pzflow.flow.Flow at 0x7f894cebba90>



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
          <td>28.131853</td>
          <td>1.142398</td>
          <td>26.800679</td>
          <td>0.179076</td>
          <td>26.026683</td>
          <td>0.080725</td>
          <td>25.377813</td>
          <td>0.074207</td>
          <td>25.034482</td>
          <td>0.104553</td>
          <td>24.714083</td>
          <td>0.175293</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.378910</td>
          <td>0.715001</td>
          <td>27.953651</td>
          <td>0.453065</td>
          <td>27.399091</td>
          <td>0.261458</td>
          <td>27.093804</td>
          <td>0.319528</td>
          <td>26.870386</td>
          <td>0.473670</td>
          <td>26.022402</td>
          <td>0.499392</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.012269</td>
          <td>0.553585</td>
          <td>26.011329</td>
          <td>0.090498</td>
          <td>24.810156</td>
          <td>0.027521</td>
          <td>23.846615</td>
          <td>0.019337</td>
          <td>23.144291</td>
          <td>0.019855</td>
          <td>22.860290</td>
          <td>0.034468</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.746045</td>
          <td>0.455039</td>
          <td>27.523916</td>
          <td>0.324786</td>
          <td>27.372713</td>
          <td>0.255872</td>
          <td>26.788902</td>
          <td>0.249602</td>
          <td>26.320026</td>
          <td>0.309255</td>
          <td>25.012275</td>
          <td>0.225202</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.703341</td>
          <td>0.440626</td>
          <td>25.748442</td>
          <td>0.071791</td>
          <td>25.422409</td>
          <td>0.047252</td>
          <td>24.784243</td>
          <td>0.043832</td>
          <td>24.357298</td>
          <td>0.057521</td>
          <td>23.617665</td>
          <td>0.067453</td>
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
          <td>26.111722</td>
          <td>0.276948</td>
          <td>26.188367</td>
          <td>0.105675</td>
          <td>26.183159</td>
          <td>0.092650</td>
          <td>25.883453</td>
          <td>0.115692</td>
          <td>26.139132</td>
          <td>0.267183</td>
          <td>25.280092</td>
          <td>0.280592</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.786062</td>
          <td>0.468888</td>
          <td>27.249003</td>
          <td>0.260176</td>
          <td>26.695807</td>
          <td>0.144743</td>
          <td>26.283647</td>
          <td>0.163390</td>
          <td>26.423749</td>
          <td>0.335881</td>
          <td>25.314531</td>
          <td>0.288525</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.240839</td>
          <td>0.258444</td>
          <td>27.431478</td>
          <td>0.268464</td>
          <td>26.386124</td>
          <td>0.178274</td>
          <td>26.239095</td>
          <td>0.289766</td>
          <td>25.625547</td>
          <td>0.369407</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.865742</td>
          <td>0.497466</td>
          <td>26.989493</td>
          <td>0.209918</td>
          <td>26.865795</td>
          <td>0.167419</td>
          <td>25.859351</td>
          <td>0.113288</td>
          <td>25.635918</td>
          <td>0.175678</td>
          <td>26.946371</td>
          <td>0.935698</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.028056</td>
          <td>0.559912</td>
          <td>26.489247</td>
          <td>0.137214</td>
          <td>25.992965</td>
          <td>0.078358</td>
          <td>25.721730</td>
          <td>0.100451</td>
          <td>25.253307</td>
          <td>0.126501</td>
          <td>24.890102</td>
          <td>0.203366</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.611495</td>
          <td>0.175108</td>
          <td>26.038557</td>
          <td>0.095916</td>
          <td>25.401505</td>
          <td>0.089749</td>
          <td>24.840631</td>
          <td>0.103643</td>
          <td>24.424528</td>
          <td>0.161049</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.672343</td>
          <td>0.839008</td>
          <td>27.359443</td>
          <td>0.293975</td>
          <td>27.064798</td>
          <td>0.363372</td>
          <td>27.476739</td>
          <td>0.821993</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.123656</td>
          <td>0.316246</td>
          <td>25.757292</td>
          <td>0.085254</td>
          <td>24.780534</td>
          <td>0.032253</td>
          <td>23.878503</td>
          <td>0.023978</td>
          <td>23.139753</td>
          <td>0.023687</td>
          <td>22.809897</td>
          <td>0.039943</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.311552</td>
          <td>0.777057</td>
          <td>27.462110</td>
          <td>0.371325</td>
          <td>27.338319</td>
          <td>0.307268</td>
          <td>26.319286</td>
          <td>0.211632</td>
          <td>26.060753</td>
          <td>0.309411</td>
          <td>24.906571</td>
          <td>0.257535</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.653545</td>
          <td>0.469338</td>
          <td>25.742244</td>
          <td>0.082472</td>
          <td>25.364049</td>
          <td>0.052865</td>
          <td>24.782503</td>
          <td>0.051927</td>
          <td>24.536259</td>
          <td>0.079348</td>
          <td>23.679820</td>
          <td>0.084337</td>
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
          <td>26.656533</td>
          <td>0.476724</td>
          <td>26.301246</td>
          <td>0.136795</td>
          <td>26.245910</td>
          <td>0.117394</td>
          <td>26.079867</td>
          <td>0.165161</td>
          <td>26.007971</td>
          <td>0.284275</td>
          <td>24.985904</td>
          <td>0.262962</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.463137</td>
          <td>0.407376</td>
          <td>26.663898</td>
          <td>0.183717</td>
          <td>26.592273</td>
          <td>0.155711</td>
          <td>26.345000</td>
          <td>0.203246</td>
          <td>26.310298</td>
          <td>0.356294</td>
          <td>25.591711</td>
          <td>0.418280</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.650210</td>
          <td>1.616366</td>
          <td>27.555200</td>
          <td>0.382138</td>
          <td>26.514578</td>
          <td>0.146904</td>
          <td>26.593437</td>
          <td>0.251907</td>
          <td>26.040126</td>
          <td>0.289569</td>
          <td>25.157034</td>
          <td>0.299753</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.188731</td>
          <td>0.290264</td>
          <td>26.648633</td>
          <td>0.167771</td>
          <td>25.682169</td>
          <td>0.118495</td>
          <td>25.722468</td>
          <td>0.227135</td>
          <td>25.220132</td>
          <td>0.320819</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.506412</td>
          <td>0.422712</td>
          <td>26.806577</td>
          <td>0.208186</td>
          <td>26.060656</td>
          <td>0.098778</td>
          <td>25.624405</td>
          <td>0.110237</td>
          <td>24.972339</td>
          <td>0.117424</td>
          <td>24.968578</td>
          <td>0.256558</td>
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
          <td>26.961530</td>
          <td>0.533663</td>
          <td>27.218214</td>
          <td>0.253725</td>
          <td>26.051971</td>
          <td>0.082557</td>
          <td>25.298630</td>
          <td>0.069196</td>
          <td>25.102994</td>
          <td>0.111015</td>
          <td>24.840081</td>
          <td>0.195022</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.747665</td>
          <td>1.421574</td>
          <td>27.412467</td>
          <td>0.264564</td>
          <td>27.150505</td>
          <td>0.334556</td>
          <td>26.398710</td>
          <td>0.329561</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.336725</td>
          <td>0.349605</td>
          <td>25.979048</td>
          <td>0.094538</td>
          <td>24.819312</td>
          <td>0.030115</td>
          <td>23.857498</td>
          <td>0.021214</td>
          <td>23.125696</td>
          <td>0.021168</td>
          <td>22.868477</td>
          <td>0.037822</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.859084</td>
          <td>0.500464</td>
          <td>27.226946</td>
          <td>0.279960</td>
          <td>26.507726</td>
          <td>0.246572</td>
          <td>25.655613</td>
          <td>0.221490</td>
          <td>25.631983</td>
          <td>0.454511</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.798950</td>
          <td>0.214336</td>
          <td>25.895374</td>
          <td>0.081826</td>
          <td>25.412304</td>
          <td>0.046897</td>
          <td>24.768933</td>
          <td>0.043306</td>
          <td>24.337637</td>
          <td>0.056608</td>
          <td>23.615904</td>
          <td>0.067448</td>
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
          <td>26.859315</td>
          <td>0.518788</td>
          <td>26.307994</td>
          <td>0.125562</td>
          <td>26.335976</td>
          <td>0.114548</td>
          <td>26.053666</td>
          <td>0.145389</td>
          <td>26.361903</td>
          <td>0.343286</td>
          <td>25.949370</td>
          <td>0.506705</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.028826</td>
          <td>0.219888</td>
          <td>26.801659</td>
          <td>0.161061</td>
          <td>26.909125</td>
          <td>0.279810</td>
          <td>26.246143</td>
          <td>0.295851</td>
          <td>25.315441</td>
          <td>0.293309</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.494020</td>
          <td>0.329646</td>
          <td>26.980903</td>
          <td>0.193481</td>
          <td>26.325354</td>
          <td>0.177902</td>
          <td>25.661945</td>
          <td>0.188247</td>
          <td>25.011814</td>
          <td>0.236148</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.641518</td>
          <td>0.451288</td>
          <td>27.112213</td>
          <td>0.255534</td>
          <td>26.740433</td>
          <td>0.168153</td>
          <td>25.762926</td>
          <td>0.117300</td>
          <td>25.654318</td>
          <td>0.199194</td>
          <td>26.630367</td>
          <td>0.834839</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.407622</td>
          <td>0.744200</td>
          <td>26.455870</td>
          <td>0.137837</td>
          <td>26.075716</td>
          <td>0.087649</td>
          <td>25.579617</td>
          <td>0.092365</td>
          <td>25.298904</td>
          <td>0.136724</td>
          <td>25.410661</td>
          <td>0.323349</td>
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
