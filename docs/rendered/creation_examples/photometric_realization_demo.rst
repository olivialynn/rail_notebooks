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

    <pzflow.flow.Flow at 0x7f04d08dcd30>



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
          <td>27.277177</td>
          <td>0.667149</td>
          <td>26.602642</td>
          <td>0.151265</td>
          <td>25.846230</td>
          <td>0.068823</td>
          <td>25.347153</td>
          <td>0.072222</td>
          <td>24.878011</td>
          <td>0.091148</td>
          <td>24.897364</td>
          <td>0.204609</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.102554</td>
          <td>0.590516</td>
          <td>27.770321</td>
          <td>0.393974</td>
          <td>28.081540</td>
          <td>0.447648</td>
          <td>27.852465</td>
          <td>0.568416</td>
          <td>26.246126</td>
          <td>0.291416</td>
          <td>25.411342</td>
          <td>0.311881</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.570367</td>
          <td>0.811510</td>
          <td>25.919644</td>
          <td>0.083490</td>
          <td>24.754360</td>
          <td>0.026214</td>
          <td>23.873717</td>
          <td>0.019785</td>
          <td>23.170919</td>
          <td>0.020308</td>
          <td>22.894907</td>
          <td>0.035538</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.497084</td>
          <td>0.773574</td>
          <td>27.757007</td>
          <td>0.389943</td>
          <td>27.090829</td>
          <td>0.202513</td>
          <td>26.502836</td>
          <td>0.196742</td>
          <td>26.110306</td>
          <td>0.260966</td>
          <td>25.486210</td>
          <td>0.331048</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.008838</td>
          <td>0.552218</td>
          <td>25.764034</td>
          <td>0.072787</td>
          <td>25.328945</td>
          <td>0.043491</td>
          <td>24.818632</td>
          <td>0.045190</td>
          <td>24.377405</td>
          <td>0.058557</td>
          <td>23.657673</td>
          <td>0.069885</td>
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
          <td>26.427250</td>
          <td>0.356221</td>
          <td>26.175135</td>
          <td>0.104461</td>
          <td>26.137587</td>
          <td>0.089012</td>
          <td>25.892270</td>
          <td>0.116583</td>
          <td>26.020450</td>
          <td>0.242402</td>
          <td>25.690092</td>
          <td>0.388407</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.567253</td>
          <td>0.397160</td>
          <td>26.934739</td>
          <td>0.200508</td>
          <td>27.175787</td>
          <td>0.217426</td>
          <td>26.491273</td>
          <td>0.194837</td>
          <td>25.701433</td>
          <td>0.185702</td>
          <td>25.984199</td>
          <td>0.485468</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.847736</td>
          <td>0.490890</td>
          <td>27.897175</td>
          <td>0.434138</td>
          <td>27.045570</td>
          <td>0.194954</td>
          <td>26.363457</td>
          <td>0.174877</td>
          <td>26.290403</td>
          <td>0.301993</td>
          <td>24.778475</td>
          <td>0.185121</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.415250</td>
          <td>0.297745</td>
          <td>26.544295</td>
          <td>0.126992</td>
          <td>25.865426</td>
          <td>0.113889</td>
          <td>25.443131</td>
          <td>0.149014</td>
          <td>25.269706</td>
          <td>0.278238</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.557759</td>
          <td>0.394265</td>
          <td>26.719144</td>
          <td>0.167095</td>
          <td>26.079971</td>
          <td>0.084609</td>
          <td>25.578736</td>
          <td>0.088598</td>
          <td>25.154583</td>
          <td>0.116104</td>
          <td>24.440766</td>
          <td>0.138729</td>
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
          <td>27.722377</td>
          <td>0.970959</td>
          <td>26.837684</td>
          <td>0.211836</td>
          <td>26.149757</td>
          <td>0.105725</td>
          <td>25.315982</td>
          <td>0.083241</td>
          <td>25.175580</td>
          <td>0.138646</td>
          <td>24.623488</td>
          <td>0.190675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.181740</td>
          <td>1.142200</td>
          <td>28.200147</td>
          <td>0.559513</td>
          <td>26.986325</td>
          <td>0.341639</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.155242</td>
          <td>0.295799</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.527443</td>
          <td>0.433036</td>
          <td>26.035268</td>
          <td>0.108749</td>
          <td>24.832595</td>
          <td>0.033765</td>
          <td>23.913221</td>
          <td>0.024709</td>
          <td>23.158006</td>
          <td>0.024063</td>
          <td>22.778223</td>
          <td>0.038839</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.330692</td>
          <td>0.786875</td>
          <td>27.485131</td>
          <td>0.378036</td>
          <td>27.811453</td>
          <td>0.444278</td>
          <td>26.638142</td>
          <td>0.275268</td>
          <td>25.626569</td>
          <td>0.216929</td>
          <td>25.666226</td>
          <td>0.467793</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.211628</td>
          <td>0.698959</td>
          <td>25.732217</td>
          <td>0.081747</td>
          <td>25.552529</td>
          <td>0.062484</td>
          <td>24.809205</td>
          <td>0.053172</td>
          <td>24.290917</td>
          <td>0.063874</td>
          <td>23.568673</td>
          <td>0.076463</td>
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
          <td>26.825195</td>
          <td>0.539619</td>
          <td>26.423771</td>
          <td>0.151984</td>
          <td>25.994679</td>
          <td>0.094252</td>
          <td>25.979023</td>
          <td>0.151516</td>
          <td>25.911215</td>
          <td>0.262761</td>
          <td>26.092407</td>
          <td>0.613038</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.467186</td>
          <td>0.829455</td>
          <td>26.782887</td>
          <td>0.203070</td>
          <td>27.623557</td>
          <td>0.363901</td>
          <td>26.417477</td>
          <td>0.215947</td>
          <td>26.501187</td>
          <td>0.413122</td>
          <td>25.450679</td>
          <td>0.375177</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.665379</td>
          <td>0.477326</td>
          <td>27.767770</td>
          <td>0.449596</td>
          <td>26.623056</td>
          <td>0.161210</td>
          <td>26.326650</td>
          <td>0.201852</td>
          <td>25.375993</td>
          <td>0.166699</td>
          <td>25.636032</td>
          <td>0.435956</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.370629</td>
          <td>0.791184</td>
          <td>27.379647</td>
          <td>0.338092</td>
          <td>26.673571</td>
          <td>0.171370</td>
          <td>26.109489</td>
          <td>0.171156</td>
          <td>25.465451</td>
          <td>0.183117</td>
          <td>26.163240</td>
          <td>0.649611</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.522647</td>
          <td>0.427966</td>
          <td>26.819954</td>
          <td>0.210527</td>
          <td>25.920252</td>
          <td>0.087320</td>
          <td>25.646543</td>
          <td>0.112386</td>
          <td>25.065064</td>
          <td>0.127266</td>
          <td>24.888401</td>
          <td>0.240189</td>
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
          <td>28.304304</td>
          <td>1.257726</td>
          <td>26.752998</td>
          <td>0.171995</td>
          <td>25.988329</td>
          <td>0.078049</td>
          <td>25.452244</td>
          <td>0.079262</td>
          <td>24.843203</td>
          <td>0.088412</td>
          <td>25.001186</td>
          <td>0.223164</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.107672</td>
          <td>0.508416</td>
          <td>27.368301</td>
          <td>0.255173</td>
          <td>28.093002</td>
          <td>0.673453</td>
          <td>26.561432</td>
          <td>0.374542</td>
          <td>25.855474</td>
          <td>0.441189</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.633958</td>
          <td>0.439703</td>
          <td>25.977058</td>
          <td>0.094373</td>
          <td>24.821759</td>
          <td>0.030180</td>
          <td>23.845721</td>
          <td>0.021002</td>
          <td>23.168987</td>
          <td>0.021965</td>
          <td>22.797547</td>
          <td>0.035524</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.916456</td>
          <td>0.521992</td>
          <td>27.327477</td>
          <td>0.303617</td>
          <td>26.484570</td>
          <td>0.241914</td>
          <td>25.798586</td>
          <td>0.249289</td>
          <td>25.333880</td>
          <td>0.361527</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.138412</td>
          <td>0.283256</td>
          <td>25.893396</td>
          <td>0.081684</td>
          <td>25.353662</td>
          <td>0.044519</td>
          <td>24.940723</td>
          <td>0.050440</td>
          <td>24.518203</td>
          <td>0.066438</td>
          <td>23.598159</td>
          <td>0.066397</td>
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
          <td>26.432077</td>
          <td>0.375758</td>
          <td>26.282564</td>
          <td>0.122825</td>
          <td>26.167225</td>
          <td>0.098844</td>
          <td>26.020391</td>
          <td>0.141285</td>
          <td>25.557422</td>
          <td>0.177324</td>
          <td>24.873891</td>
          <td>0.216821</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.037374</td>
          <td>0.568975</td>
          <td>27.124148</td>
          <td>0.237970</td>
          <td>27.314827</td>
          <td>0.247778</td>
          <td>26.361267</td>
          <td>0.177491</td>
          <td>25.812532</td>
          <td>0.207134</td>
          <td>25.283099</td>
          <td>0.285746</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.016888</td>
          <td>0.223647</td>
          <td>27.058086</td>
          <td>0.206440</td>
          <td>26.557879</td>
          <td>0.216336</td>
          <td>25.432219</td>
          <td>0.154837</td>
          <td>25.760104</td>
          <td>0.428398</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.604964</td>
          <td>0.378833</td>
          <td>26.570764</td>
          <td>0.145427</td>
          <td>26.201093</td>
          <td>0.171048</td>
          <td>25.372687</td>
          <td>0.156870</td>
          <td>25.364277</td>
          <td>0.334491</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.095839</td>
          <td>0.600768</td>
          <td>26.517848</td>
          <td>0.145388</td>
          <td>26.050509</td>
          <td>0.085725</td>
          <td>25.639439</td>
          <td>0.097345</td>
          <td>25.059307</td>
          <td>0.111056</td>
          <td>24.816660</td>
          <td>0.198728</td>
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
