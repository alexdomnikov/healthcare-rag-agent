"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function DisclaimerGate({ children }: { children: React.ReactNode }) {
  const [accepted, setAccepted] = useState(false);
  const [checked, setChecked] = useState(false);

  function accept() {
    setAccepted(true);
  }

  if (accepted) {
    return <>{children}</>;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="max-w-2xl w-full bg-white rounded-lg shadow-xl max-h-[90dvh] flex flex-col">
        <div className="px-6 py-4 border-b border-neutral-200 shrink-0">
          <h2 className="text-lg font-semibold text-neutral-900">
            Important Notice: Please Read Before Continuing
          </h2>
        </div>

        <div className="px-6 py-4 overflow-y-auto text-sm text-neutral-800 space-y-3 leading-relaxed">
          <p className="font-semibold text-red-700">
            THIS IS A TECHNICAL DEMONSTRATION ONLY. IT IS NOT A MEDICAL,
            LEGAL, REGULATORY, COMPLIANCE, OR PROFESSIONAL ADVICE TOOL.
          </p>

          <p>
            This application is a personal portfolio / educational project that
            uses a large language model (LLM) to generate responses based on
            retrieved documents. The output is{" "}
            <strong>artificially generated</strong>, may be{" "}
            <strong>inaccurate, incomplete, outdated, or entirely fabricated</strong>{" "}
            (&ldquo;hallucinated&rdquo;), and must not be relied upon for any
            decision of any kind.
          </p>

          <p className="font-semibold">By using this site, you acknowledge and agree that:</p>

          <ul className="list-disc pl-5 space-y-1.5">
            <li>
              The information provided is <strong>NOT medical advice</strong>,{" "}
              <strong>NOT legal advice</strong>, <strong>NOT regulatory or
              compliance advice</strong>, and <strong>NOT a substitute for
              consultation with a licensed physician, attorney, compliance
              officer, or other qualified professional</strong>.
            </li>
            <li>
              No physician-patient, attorney-client, or other professional
              relationship is created by your use of this site.
            </li>
            <li>
              You will <strong>NOT</strong> use any output from this site to
              make, inform, or influence any clinical, diagnostic, treatment,
              prescribing, insurance, enrollment, regulatory, compliance,
              financial, or legal decision.
            </li>
            <li>
              The site is provided <strong>&ldquo;AS IS&rdquo;</strong> and{" "}
              <strong>&ldquo;AS AVAILABLE&rdquo;</strong> with{" "}
              <strong>no warranties of any kind</strong>, express or implied,
              including but not limited to warranties of accuracy,
              completeness, timeliness, merchantability, fitness for a
              particular purpose, or non-infringement.
            </li>
            <li>
              The developer, contributors, and any associated parties{" "}
              <strong>disclaim all liability</strong> for any direct, indirect,
              incidental, consequential, special, exemplary, or punitive
              damages of any kind arising from or related to your use of, or
              reliance on, this site or its output, to the fullest extent
              permitted by law.
            </li>
            <li>
              You assume <strong>all risk</strong> associated with your use of
              this site and any actions you take based on its output.
            </li>
            <li>
              For authoritative information, you must consult the primary
              sources directly (e.g., the Code of Federal Regulations, the
              FDA, CMS) and a qualified professional.
            </li>
            <li>
              <strong>
                If you are experiencing a medical emergency, call 911 or your
                local emergency number immediately. Do not use this site.
              </strong>
            </li>
          </ul>

          <p className="text-xs text-neutral-500 pt-2 border-t border-neutral-200">
            This notice does not create any contract, warranty, or duty owed to
            you. If you do not accept these terms in full, do not use the site.
          </p>
        </div>

        <div className="px-6 py-4 border-t border-neutral-200 shrink-0 space-y-3">
          <label className="flex items-start gap-2 text-sm text-neutral-800 cursor-pointer">
            <input
              type="checkbox"
              checked={checked}
              onChange={(e) => setChecked(e.target.checked)}
              className="mt-0.5 h-4 w-4 cursor-pointer"
            />
            <span>
              I have read and understood this notice. I accept all terms above
              and agree that I will not rely on this site for any medical,
              legal, regulatory, or professional decision.
            </span>
          </label>
          <Button
            onClick={accept}
            disabled={!checked}
            className="w-full"
          >
            I Agree, Enter Demo
          </Button>
        </div>
      </div>
    </div>
  );
}
