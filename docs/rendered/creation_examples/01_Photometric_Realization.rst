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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f4d382d4df0>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.099593  0.053469  
    1      25.391064  0.195476  0.139703  
    2      24.304707  0.010792  0.010370  
    3      25.291103  0.003372  0.002707  
    4      25.096743  0.005953  0.005870  
    ...          ...       ...       ...  
    99995  24.737946  0.111545  0.094632  
    99996  24.224169  0.045760  0.035236  
    99997  25.613836  0.077018  0.054647  
    99998  25.274899  0.031851  0.030533  
    99999  25.699642  0.006591  0.004155  
    
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
          <td>1.398944</td>
          <td>29.338634</td>
          <td>2.059149</td>
          <td>26.756394</td>
          <td>0.172473</td>
          <td>25.997194</td>
          <td>0.078652</td>
          <td>25.189231</td>
          <td>0.062794</td>
          <td>24.634449</td>
          <td>0.073532</td>
          <td>24.100707</td>
          <td>0.103234</td>
          <td>0.099593</td>
          <td>0.053469</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.656596</td>
          <td>0.425277</td>
          <td>27.608124</td>
          <td>0.347168</td>
          <td>26.632564</td>
          <td>0.137066</td>
          <td>26.200959</td>
          <td>0.152231</td>
          <td>25.891854</td>
          <td>0.217889</td>
          <td>26.063925</td>
          <td>0.514880</td>
          <td>0.195476</td>
          <td>0.139703</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.976810</td>
          <td>2.622323</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.170153</td>
          <td>0.148259</td>
          <td>25.155072</td>
          <td>0.116154</td>
          <td>24.231417</td>
          <td>0.115710</td>
          <td>0.010792</td>
          <td>0.010370</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.859884</td>
          <td>0.495319</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.073266</td>
          <td>0.199548</td>
          <td>26.185536</td>
          <td>0.150230</td>
          <td>25.289198</td>
          <td>0.130495</td>
          <td>24.862829</td>
          <td>0.198763</td>
          <td>0.003372</td>
          <td>0.002707</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.413423</td>
          <td>0.352378</td>
          <td>26.071925</td>
          <td>0.095440</td>
          <td>26.032474</td>
          <td>0.081139</td>
          <td>25.598939</td>
          <td>0.090187</td>
          <td>25.465338</td>
          <td>0.151880</td>
          <td>25.459844</td>
          <td>0.324186</td>
          <td>0.005953</td>
          <td>0.005870</td>
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
          <td>0.389450</td>
          <td>27.612066</td>
          <td>0.833648</td>
          <td>26.360701</td>
          <td>0.122777</td>
          <td>25.358960</td>
          <td>0.044664</td>
          <td>24.994422</td>
          <td>0.052824</td>
          <td>24.831597</td>
          <td>0.087502</td>
          <td>24.707979</td>
          <td>0.174387</td>
          <td>0.111545</td>
          <td>0.094632</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.393374</td>
          <td>1.319517</td>
          <td>27.506195</td>
          <td>0.320236</td>
          <td>26.163399</td>
          <td>0.091055</td>
          <td>25.136154</td>
          <td>0.059906</td>
          <td>24.842632</td>
          <td>0.088356</td>
          <td>24.435169</td>
          <td>0.138061</td>
          <td>0.045760</td>
          <td>0.035236</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.682685</td>
          <td>0.433788</td>
          <td>26.659193</td>
          <td>0.158766</td>
          <td>26.426199</td>
          <td>0.114606</td>
          <td>26.475095</td>
          <td>0.192200</td>
          <td>25.953683</td>
          <td>0.229382</td>
          <td>26.225288</td>
          <td>0.578642</td>
          <td>0.077018</td>
          <td>0.054647</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.824523</td>
          <td>0.952592</td>
          <td>26.209298</td>
          <td>0.107624</td>
          <td>25.890792</td>
          <td>0.071592</td>
          <td>25.756264</td>
          <td>0.103534</td>
          <td>25.920084</td>
          <td>0.223070</td>
          <td>25.721286</td>
          <td>0.397877</td>
          <td>0.031851</td>
          <td>0.030533</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.685395</td>
          <td>0.873543</td>
          <td>27.097190</td>
          <td>0.229605</td>
          <td>26.615735</td>
          <td>0.135089</td>
          <td>26.731372</td>
          <td>0.238045</td>
          <td>26.186321</td>
          <td>0.277643</td>
          <td>25.445329</td>
          <td>0.320460</td>
          <td>0.006591</td>
          <td>0.004155</td>
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
          <td>1.398944</td>
          <td>26.462404</td>
          <td>0.411780</td>
          <td>26.498702</td>
          <td>0.162058</td>
          <td>26.165022</td>
          <td>0.109420</td>
          <td>25.172517</td>
          <td>0.074974</td>
          <td>24.820622</td>
          <td>0.104010</td>
          <td>24.032386</td>
          <td>0.117316</td>
          <td>0.099593</td>
          <td>0.053469</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.242054</td>
          <td>0.317766</td>
          <td>26.926009</td>
          <td>0.223849</td>
          <td>26.014598</td>
          <td>0.167174</td>
          <td>26.658923</td>
          <td>0.499987</td>
          <td>26.458626</td>
          <td>0.827647</td>
          <td>0.195476</td>
          <td>0.139703</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.307508</td>
          <td>0.745502</td>
          <td>27.732903</td>
          <td>0.433691</td>
          <td>28.171619</td>
          <td>0.548184</td>
          <td>26.109385</td>
          <td>0.165895</td>
          <td>25.261135</td>
          <td>0.149286</td>
          <td>24.319378</td>
          <td>0.147229</td>
          <td>0.010792</td>
          <td>0.010370</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.389715</td>
          <td>0.786928</td>
          <td>27.694161</td>
          <td>0.420980</td>
          <td>27.667543</td>
          <td>0.375214</td>
          <td>26.137144</td>
          <td>0.169803</td>
          <td>25.732867</td>
          <td>0.222407</td>
          <td>25.717878</td>
          <td>0.458528</td>
          <td>0.003372</td>
          <td>0.002707</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.775042</td>
          <td>0.234574</td>
          <td>26.049937</td>
          <td>0.107971</td>
          <td>25.978980</td>
          <td>0.091036</td>
          <td>25.730220</td>
          <td>0.119649</td>
          <td>25.410977</td>
          <td>0.169640</td>
          <td>25.007881</td>
          <td>0.262433</td>
          <td>0.005953</td>
          <td>0.005870</td>
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
          <td>0.389450</td>
          <td>26.150322</td>
          <td>0.325911</td>
          <td>26.347137</td>
          <td>0.144100</td>
          <td>25.379552</td>
          <td>0.055518</td>
          <td>25.071620</td>
          <td>0.069580</td>
          <td>24.762463</td>
          <td>0.100234</td>
          <td>24.890464</td>
          <td>0.246469</td>
          <td>0.111545</td>
          <td>0.094632</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.226195</td>
          <td>0.708081</td>
          <td>26.799882</td>
          <td>0.206238</td>
          <td>26.052535</td>
          <td>0.097646</td>
          <td>25.166454</td>
          <td>0.073377</td>
          <td>24.973083</td>
          <td>0.116987</td>
          <td>24.080976</td>
          <td>0.120462</td>
          <td>0.045760</td>
          <td>0.035236</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.431591</td>
          <td>0.815506</td>
          <td>26.892908</td>
          <td>0.224652</td>
          <td>26.340646</td>
          <td>0.126693</td>
          <td>26.332487</td>
          <td>0.203268</td>
          <td>26.127635</td>
          <td>0.311276</td>
          <td>25.296936</td>
          <td>0.335816</td>
          <td>0.077018</td>
          <td>0.054647</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.092127</td>
          <td>0.304315</td>
          <td>26.302801</td>
          <td>0.134840</td>
          <td>26.072438</td>
          <td>0.099132</td>
          <td>25.830085</td>
          <td>0.130897</td>
          <td>25.934785</td>
          <td>0.263497</td>
          <td>25.398146</td>
          <td>0.359804</td>
          <td>0.031851</td>
          <td>0.030533</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.140329</td>
          <td>0.315568</td>
          <td>26.588313</td>
          <td>0.171706</td>
          <td>26.533496</td>
          <td>0.147465</td>
          <td>26.042871</td>
          <td>0.156690</td>
          <td>26.094236</td>
          <td>0.298974</td>
          <td>26.657949</td>
          <td>0.879894</td>
          <td>0.006591</td>
          <td>0.004155</td>
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
          <td>1.398944</td>
          <td>27.869753</td>
          <td>1.017282</td>
          <td>26.791652</td>
          <td>0.189832</td>
          <td>25.896546</td>
          <td>0.077846</td>
          <td>25.178436</td>
          <td>0.067543</td>
          <td>24.710020</td>
          <td>0.085019</td>
          <td>23.962858</td>
          <td>0.099181</td>
          <td>0.099593</td>
          <td>0.053469</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.847724</td>
          <td>1.114313</td>
          <td>27.715098</td>
          <td>0.469869</td>
          <td>26.844724</td>
          <td>0.215441</td>
          <td>26.171778</td>
          <td>0.196899</td>
          <td>25.780836</td>
          <td>0.258897</td>
          <td>25.535255</td>
          <td>0.443839</td>
          <td>0.195476</td>
          <td>0.139703</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.564402</td>
          <td>2.254910</td>
          <td>29.079572</td>
          <td>0.978272</td>
          <td>28.183018</td>
          <td>0.483611</td>
          <td>26.214494</td>
          <td>0.154248</td>
          <td>25.032570</td>
          <td>0.104536</td>
          <td>24.246233</td>
          <td>0.117396</td>
          <td>0.010792</td>
          <td>0.010370</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.858750</td>
          <td>0.421673</td>
          <td>27.492621</td>
          <td>0.282177</td>
          <td>26.622815</td>
          <td>0.217565</td>
          <td>25.531697</td>
          <td>0.160777</td>
          <td>25.542301</td>
          <td>0.346102</td>
          <td>0.003372</td>
          <td>0.002707</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.450859</td>
          <td>0.362970</td>
          <td>26.096856</td>
          <td>0.097588</td>
          <td>25.868340</td>
          <td>0.070217</td>
          <td>25.690025</td>
          <td>0.097746</td>
          <td>25.347121</td>
          <td>0.137257</td>
          <td>24.670365</td>
          <td>0.168978</td>
          <td>0.005953</td>
          <td>0.005870</td>
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
          <td>0.389450</td>
          <td>26.838560</td>
          <td>0.526183</td>
          <td>26.658640</td>
          <td>0.176819</td>
          <td>25.456190</td>
          <td>0.055356</td>
          <td>25.201141</td>
          <td>0.072523</td>
          <td>24.889588</td>
          <td>0.104480</td>
          <td>24.732608</td>
          <td>0.201935</td>
          <td>0.111545</td>
          <td>0.094632</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.897641</td>
          <td>0.515881</td>
          <td>26.848165</td>
          <td>0.189890</td>
          <td>26.031471</td>
          <td>0.082890</td>
          <td>25.090632</td>
          <td>0.058902</td>
          <td>24.951213</td>
          <td>0.099368</td>
          <td>23.999481</td>
          <td>0.096664</td>
          <td>0.045760</td>
          <td>0.035236</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.824879</td>
          <td>0.499189</td>
          <td>26.619559</td>
          <td>0.161066</td>
          <td>26.381941</td>
          <td>0.116656</td>
          <td>26.450145</td>
          <td>0.199225</td>
          <td>25.898107</td>
          <td>0.231082</td>
          <td>25.940498</td>
          <td>0.493892</td>
          <td>0.077018</td>
          <td>0.054647</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.666798</td>
          <td>0.431961</td>
          <td>26.066565</td>
          <td>0.096075</td>
          <td>25.978690</td>
          <td>0.078400</td>
          <td>26.006052</td>
          <td>0.130440</td>
          <td>25.656061</td>
          <td>0.180966</td>
          <td>25.375585</td>
          <td>0.306845</td>
          <td>0.031851</td>
          <td>0.030533</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.450125</td>
          <td>0.362748</td>
          <td>26.602866</td>
          <td>0.151346</td>
          <td>26.403985</td>
          <td>0.112455</td>
          <td>26.151737</td>
          <td>0.145993</td>
          <td>25.563533</td>
          <td>0.165251</td>
          <td>25.717825</td>
          <td>0.396966</td>
          <td>0.006591</td>
          <td>0.004155</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
