# Qualitative examples (auto-extracted; numbers in nats)

## A_alignment_gain: google_gemini-3-1-pro-preview on Democracy :: Banking (item E0)

valid mapping=True (failure channel=ok); panel: faithful=True (unanimous=True), coherent=True (unanimous=True)

Source path:
```
  Democracy --aggregates--> Ballots
  Ballots --empower--> Parliaments
  Parliaments --approve--> Laws
  Laws --bind--> Citizens
```
Target path:
```
  Banking --aggregates--> Deposits
  Deposits --empower--> Banks
  Banks --approve--> Loans
  Loans --bind--> Borrowers
```
Cost of the target path's entity tokens (log p): anchor only=-22.3, anchor + skeleton=-20.4, true source path=-3.9, donor entities=-45.2
  alignment gain (true source vs skeleton) = 16.6; donor entities vs skeleton = -24.7

Projected 'Fractional Reserve' -> invention 'Fractional Mandate'

Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:
  Banks --utilize--> Fractional Reserve: gain=-8.4, plausibility=-3.90
  Fractional Reserve --leverages--> Deposits: gain=-18.7, plausibility=-1.93
  Fractional Reserve --multiplies--> Loans: gain=-7.4, plausibility=-3.74
Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:
  Parliaments --utilize--> Fractional Mandate: gain=+6.7, plausibility=-3.28
  Fractional Mandate --leverages--> Ballots: gain=+16.5, plausibility=-1.65
  Fractional Mandate --multiplies--> Laws: gain=+27.3, plausibility=-4.03

  judge anthropic/claude-haiku-4.5: faithful=True coherent=True
  judge openai/gpt-5.4: faithful=True coherent=True
  judge openai/o3: faithful=True coherent=True

## B_faithful_invention: x-ai_grok-4-6 on The Roman Empire :: Crystals (item E13)

valid mapping=True (failure channel=ok); panel: faithful=True (unanimous=True), coherent=True (unanimous=True)

Source path:
```
  The Roman Empire --requires--> legions
  legions --requires--> training
```
Target path:
```
  Crystals --requires--> nucleation
  nucleation --requires--> supersaturation
```
Cost of the target path's entity tokens (log p): anchor only=-19.5, anchor + skeleton=-23.6, true source path=-26.8, donor entities=-24.4
  alignment gain (true source vs skeleton) = -3.1; donor entities vs skeleton = -0.8

Projected 'energy barrier' -> invention 'levy barrier'

Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:
  nucleation --overcomes--> energy barrier: gain=+15.1, plausibility=-7.95
Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:
  legions --overcomes--> levy barrier: gain=+40.8, plausibility=-8.09

  judge anthropic/claude-haiku-4.5: faithful=True coherent=True
  judge openai/gpt-5.4: faithful=True coherent=True
  judge openai/o3: faithful=True coherent=True

## C_unfaithful_invention: z-ai_glm-4-5-air on The blue whale :: The mattress (item E26)

valid mapping=True (failure channel=ok); panel: faithful=False (unanimous=True), coherent=True (unanimous=True)

Source path:
```
  The blue whale --hosts--> barnacle-covered skin
  barnacle-covered skin --creates--> specialized ecosystem
  specialized ecosystem --acts as--> environmental indicators
```
Target path:
```
  The mattress --hosts--> allergen-trapping surface
  allergen-trapping surface --creates--> unhealthy environment
  unhealthy environment --acts as--> allergy indicators
```
Cost of the target path's entity tokens (log p): anchor only=-126.1, anchor + skeleton=-99.3, true source path=-50.7, donor entities=-128.6
  alignment gain (true source vs skeleton) = 48.7; donor entities vs skeleton = -29.2

Projected 'whale skin microbiome' -> invention 'allergen monitoring system'

Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:
  whale skin microbiome --contains--> diverse organisms: gain=-5.3, plausibility=-3.11
  whale skin microbiome --indicates--> ocean health: gain=-20.8, plausibility=-2.73
  whale skin microbiome --supports--> scientific research: gain=-7.6, plausibility=-2.97
Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:
  allergen monitoring system --detects--> various particles: gain=+14.8, plausibility=-5.30
  allergen monitoring system --reflects--> indoor air quality: gain=-5.0, plausibility=-2.09
  allergen monitoring system --enables--> personalized health advice: gain=+2.2, plausibility=-3.66

  judge anthropic/claude-haiku-4.5: faithful=False coherent=True
  judge openai/gpt-5.4: faithful=False coherent=True
  judge openai/o3: faithful=False coherent=True

## D_faithful_but_incoherent: google_gemini-2-5-pro on Hinduism :: Gravity (item E3)

valid mapping=True (failure channel=ok); panel: faithful=True (unanimous=True), coherent=False (unanimous=False)

Source path:
```
  Hinduism --includes the principle of--> Dharma
  Dharma --is maintained by--> Vishnu
  Vishnu --produces--> an Avatar
```
Target path:
```
  Gravity --includes the principle of--> a Geodesic
  a Geodesic --is maintained by--> Spacetime Curvature
  Spacetime Curvature --produces--> a Gravitational Wave
```
Cost of the target path's entity tokens (log p): anchor only=-76.9, anchor + skeleton=-74.8, true source path=-80.5, donor entities=-87.8
  alignment gain (true source vs skeleton) = -5.7; donor entities vs skeleton = -12.9

Projected 'Dashavatara' -> invention 'Standard Chirps'

Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:
  Dashavatara --is a canonical set of--> an Avatar: gain=+8.0, plausibility=-4.17
  Dashavatara --sequentially restores--> Dharma: gain=-8.8, plausibility=-4.35
  Dashavatara --are manifestations of--> Vishnu: gain=-8.4, plausibility=-1.81
Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:
  Standard Chirps --is a canonical set of--> a Gravitational Wave: gain=+55.9, plausibility=-4.31
  Standard Chirps --sequentially restores--> a Geodesic: gain=+37.9, plausibility=-3.77
  Standard Chirps --are manifestations of--> Spacetime Curvature: gain=+21.9, plausibility=-2.29

  judge anthropic/claude-haiku-4.5: faithful=True coherent=True
  judge openai/gpt-5.4: faithful=True coherent=False
  judge openai/o3: faithful=True coherent=False

## E_alignment_blind_to_truth: z-ai_glm-4-5-air on Don Quixote :: Pi (item E23)

valid mapping=False (failure channel=factual); panel: faithful=False (unanimous=False), coherent=True (unanimous=True)

Source path:
```
  Don Quixote --projects--> Ideals onto reality
  Ideals onto reality --creates--> Delusional perceptions
  Delusional perceptions --result in--> Misguided actions
  Misguided actions --lead to--> Unintended consequences
```
Target path:
```
  Pi --projects--> Perfect circles onto geometry
  Perfect circles onto geometry --creates--> Idealized measurements
  Idealized measurements --result in--> Theoretical predictions
  Theoretical predictions --lead to--> Unexpected applications
```
Cost of the target path's entity tokens (log p): anchor only=-137.6, anchor + skeleton=-129.4, true source path=-48.7, donor entities=-118.4
  alignment gain (true source vs skeleton) = 80.7; donor entities vs skeleton = 11.0

Projected 'projection of ideals onto reality' -> invention 'projection of perfect forms onto geometry'

Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:
  projection of ideals onto reality --reveals--> Human nature: gain=+15.4, plausibility=-4.14
Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:
  projection of perfect forms onto geometry --reveals--> Mathematical nature: gain=+40.3, plausibility=-4.30

  judge anthropic/claude-haiku-4.5: faithful=False coherent=True
  judge openai/gpt-5.4: faithful=False coherent=True
  judge openai/o3: faithful=True coherent=True
